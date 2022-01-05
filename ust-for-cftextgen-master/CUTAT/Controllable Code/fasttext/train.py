import fasttext

DATASET = "yelp"

model = fasttext.train_supervised(f'{DATASET}.train.processed.txt')
print(model.words)
print(model.labels)


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

print_results(*model.test(f'{DATASET}.valid.processed.txt'))
model.save_model(f"{DATASET}_model.bin")