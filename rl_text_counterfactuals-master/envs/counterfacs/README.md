# Counterfactual-Text-Gym_Environment
This is an implementation of the counterfactual text generation game.
Given a movie review that has a sentiment of either Positive or Negative,
find the smallest word level change that gets you to a movie review of the other class.
"Smallest" here is measured both by how many words get changed and how (measured in cosine similarity distance)
Additionally, we prefer a larger percentage of words to get changed slightly than one word getting changed dramatically 
to mimic more realistic human like counterfactuals ( the data set we use contains human generated counterfactuals )
