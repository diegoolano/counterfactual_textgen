import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    warnings.filterwarnings("ignore",category=FutureWarning)
    from typing import Iterable
    import numpy as np
    import collections
    import itertools
    import time
    import tensorflow as tf
    from scipy.special import softmax
    #>>> xn = np.array([-1.200, -.0033])
    #>>> softmax(xn)
    #array([0.23206279, 0.76793721])


class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """

        start_time = time.time()
        self.state_dims = state_dims
        self.num_actions = num_actions
        scope = "policy_approx"

        self.state = tf.placeholder(dtype=tf.float32, shape=(None,self.state_dims), name="state")
        self.action = tf.placeholder(dtype=tf.int32, name="action")
        self.target = tf.placeholder(dtype=tf.float32, name="target")

        hiddensize = 32

        self.l1 = tf.layers.dense(self.state, units=hiddensize, activation=tf.nn.relu)
        self.l2 = tf.layers.dense(self.l1, units=hiddensize, activation=tf.nn.relu)
        self.logits = tf.layers.dense(self.l2, units=num_actions, activation=None)

        self.action_probs = tf.squeeze(tf.nn.softmax(self.logits))
        self.picked_action_prob = tf.gather(self.action_probs, self.action)

        # Loss and train op
        self.loss = -tf.log(self.picked_action_prob) * self.target

        self.alpha = 0.0003  # <= 3 * 10^-4
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.alpha,beta1=self.beta1,beta2=self.beta2)
        self.train_op = self.optimizer.minimize(
            self.loss, global_step=tf.contrib.framework.get_global_step())

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def __call__(self,s) -> int:
        debug = False
        s = np.asarray(s).reshape(1,self.state_dims)
        action_probs = self.sess.run(self.action_probs, { self.state: s })
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        if debug:
            print("For state",s,"action probs:",action_probs,"and action selected=",action)
        return action

    def update(self, s, a, gamma_t, delta, sess=None):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        debug = True
        s = np.asarray(s).reshape(1,self.state_dims)
        feed_dict = { self.state: s, self.target: delta, self.action: a  }
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict)

        if debug:
            print("UPDATE State s With Action",a,"and Gamma",gamma_t,"and Target",delta,"gives loss",loss)
        return loss

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.state_dims = state_dims
        self.b = state_dims
        self.state = tf.placeholder(shape=(None, state_dims), dtype=tf.float32,name="state")
        self.target = tf.placeholder(dtype=tf.float32, name="target")

        # The idea here is our Value Approximator takes in our current state: 
        # ([current word embedding] + [current context embedding ] + [one hot embedding of pos ])
        # and returns value of the given state 
        # It updates its estimates after each episode based on the states and cumulated rewards obtained over the episodes trajectory  

        # The input dim is 1556 so its possible the bottle neck here is too great of a change and this net needs to be larger.

        hiddensize = 32
        self.l1 = tf.layers.dense(self.state, units=hiddensize, activation=tf.nn.relu)
        self.l2 = tf.layers.dense(self.l1, units=hiddensize, activation=tf.nn.relu)
        self.output_layer = tf.layers.dense(self.l2, units=1, activation=None)

        self.value_estimate = tf.squeeze(self.output_layer)
        self.loss = tf.squared_difference(self.value_estimate, self.target)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=alpha)
        self.train_op = self.optimizer.minimize(
            self.loss, global_step=tf.contrib.framework.get_global_step())

        self.vsess = tf.Session()
        self.vsess.run(tf.global_variables_initializer())

    def __call__(self,s, sess=None) -> float:
        debug = True
        if s.shape[0] == self.state_dims:
            s = np.expand_dims(s, axis=0)

        ret = self.vsess.run(self.value_estimate, { self.state: s })
        if debug:
            print("V(s) returns", ret)

        return ret

    def update(self, s, G, sess=None):
        debug = True
        if s.shape[0] == self.state_dims:
            s = np.expand_dims(s, axis=0)

        feed_dict = { self.state: s, self.target: G }
        _, loss = self.vsess.run([self.train_op, self.loss], feed_dict)

        if debug:
            print("UPDATE value approx with gives loss",loss)

        return loss


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    debug = True
    start_time = time.time()
    if debug:
        print("session initialize: ", time.time() - start_time)

    # initialize state-value weights to zero
    if V.b == 0:
        #no baseline
        with_baseline = 0
    else:
        with_baseline = 1

    print("With Baseline:",with_baseline," V type",type(V)) #With Baseline: 0  V type <class 'reinforce.Baseline'>

    # Keeps track of useful statistics
    final_stats = {"episode_lengths":np.zeros(num_episodes), 
                   "episode_rewards":np.zeros(num_episodes),
                   "episode_cf_found":np.zeros(num_episodes),
                   "episode_time_len":np.zeros(num_episodes),
                   "episode_start_end_sent_dist":[],            
                   "episode_trajs":[],
                   "episode_trajs_loss":[]}
    
    # trajs is so [ (prior_word,prior_pos), action, (new_word, new_pos) ] 
    

    for i_episode in range(num_episodes):
        start_ep = time.time()

        # Reset the environment and get current movie review
        state = env.reset()
        if debug:
            print("After reset:",state)   
            # state takes format:
            #[ currentword, currentstence, pos, embedding version, (prior word, pos) ]
            # ex. ['consider', 'I consider this movie a blunder ... that happens. Have you noticed that?', 'VERB', array([-0.140, -0.539,  2.355, ...,  0.]), ('priorword','VERB')   ]

        episode = []
        sub_stay = {0:'SUBWORD', 1:'NO CHANGE'}
        for t in itertools.count():
            # Take a step
            action = pi(state[3])  
            next_state, reward, done, info = env.step(action)     #action 0 means substitute, 1 means do nothing
            
            transaction = [state,action,reward,next_state,done,info]
            episode.append(transaction)
            #info is for debug purposes and looks like [ prior_word, self.changed_to, self.distance_from_original, self.softmaxed, goal_reached ]

            if debug:
                labprobs = 0 if str(type(transaction[5][3])) == "<class 'int'>" else [round(pr,4) for pr in transaction[5][3][0]]
                orig_to_now_dist = 0 if str(type(transaction[5][2])) == "<class 'int'>" else round(transaction[5][2][0],4)
                prior_pair = transaction[5][0]
                now_pair = transaction[5][1]
                goal_reached = transaction[5][4]
                print("Ep",i_episode,"Step",t,"Act",sub_stay[action],"Reward",reward,"Dist",orig_to_now_dist,"LabProb:",labprobs,"--(prior -> new)--",prior_pair," --> ",now_pair,"Goal:",goal_reached)
                if t < -1:
                    break

            final_stats["episode_rewards"][i_episode] += reward
            final_stats["episode_lengths"][i_episode] = t

            #debugging 
            if done:
                end_collect = time.time()
                final_stats["episode_cf_found"][i_episode] = transaction[5][4]

                orig_to_now_dist = 0 if str(type(transaction[5][2])) == "<class 'int'>" else round(transaction[5][2][0],4)
                if debug:
                    print("\rDONE AT Step {} @ Episode {}/{} ({})".format(
                        t, i_episode + 1, num_episodes, final_stats["episode_rewards"][i_episode - 1]))#, end="")
                    print("\rEpisode collection for ep{}=({})".format( i_episode, end_collect - start_ep))
                break

            state = next_state

        if debug:
            print("************************")
            print("Now look at episode:")
            """
            print("Starting at Input sentence:",episode[0][0][1])
            for e in range(len(episode)):
                print(e,"In State",episode[e][0][0],"(",episode[e][0][2],") took Action",sub_stay[episode[e][1]],"got Reward",episode[e][2])
                if episode[e][1] != 1:
                    print("------ (prior -> new) ---",episode[e][5][0]," --> ",episode[e][5][1])
                    print("--was--",episode[e][0][1])
                    print("--now--",episode[e][3][1])
            """
            env.render()
        final_stats["episode_start_end_sent_dist"].append((episode[0][0][1],episode[-1][0][1],orig_to_now_dist))            
    

        # Go through the episode and make policy updates

        # for analysis:
        # trajs is so [ (prior_word,prior_pos), action, (new_word, new_pos) ] 
        cur_trajs, cur_trajs_loss = [],[]
        for t, transition in enumerate(episode):
            # return after this time step
            G = sum(gamma**i * tr[2] for i, tr in enumerate(episode[t:]))

            # calculate baseline/advantage
            #s, a = transition["state"].reshape(1,4), transition["action"]
            s = transition[0][3]  #use only embedding
            a = transition[1]

            if with_baseline == 1:
                    
                baseline_value = V(s) 
                delta = G - baseline_value
                #if debug:
                #    print("in with baseline G",G,"basline",baseline_value,"delta",delta)

                # Update our value approx
                if debug:
                    print("in with baseline with STATE",transition[0][0],transition[0][1][0:50],transition[0][2],"ACTION:",a,"G",G,"Baseline V",baseline_value,"delta",delta)

                V.update(s, delta)                  #update w  <-- should this be G
                cur_loss = pi.update(s, a, gamma, delta )   #update theta
            else:
                if debug:
                    print("in without baseline with STATE",transition[0][0],transition[0][1][0:50],transition[0][2],"ACTION:",a,"G",G)
                # Update our policy estimator
                cur_loss = pi.update(s, a, gamma, G )   #HERE pi yells at update but V doesn't?? 

            prior_pair, now_pair, r = transition[5][0], transition[5][1], transition[2]
            cur_trajs.append([prior_pair, a, now_pair, r, G])
            cur_trajs_loss.append(cur_loss)

        #update trajs, losses, timees
        end_ep = time.time()
        final_stats["episode_trajs"].append(cur_trajs)
        final_stats["episode_trajs_loss"].append(cur_trajs_loss[0])
        final_stats["episode_time_len"][i_episode] = end_ep - start_ep 
        if debug:
            print("\r--- Episode time for ep{}=({})".format( i_episode, end_ep - start_ep))

    #return list that includes the G_0 for every episodes.
    print("*********************^*^*^*^*^*^*^*^*^*^*^**^*^^*")
    print("DONE !!!")
    print(final_stats)
    print("*********************^*^*^*^*^*^*^*^*^*^*^**^*^^*")
    end_time= time.time()
    print("Elapsed Time", end_time - start_time)
    return final_stats
