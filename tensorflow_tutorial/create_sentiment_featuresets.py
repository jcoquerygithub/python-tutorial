import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
# max line read in file to gentle memory footprintm
hm_lines = 10000000


def create_lexicon(pos, neg):
    lexicon = []
    for file in [pos, neg]:
        with open(file,'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)

    # raw lexicon: 13452
    w_counts_raw = Counter(lexicon)
    print('Raw lexicon count:             ', len(w_counts_raw))
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    # after lemmatize: 12418
    w_counts = Counter(lexicon)
    # w_counts = {'the': 3232O4, 'and': 23499}
    print('Lexicon lemmatized count:      ', len(w_counts))

    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    # after filtering data (huge filtering!!!!): 434
    print('Custom filtered lexicon count: ', len(l2))
    return l2


def sample_handling(sample, lexicon, classification):
    featureset = []

    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])

    return featureset


def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling(pos, lexicon, [1, 0])
    features += sample_handling(neg, lexicon, [0, 1])
    random.shuffle(features)

    features = np.array(features)

    testing_size = int(test_size * len(features))
    # array of number of lexicon word occurrence in one sentence
    train_x = list(features[:, 0][:-testing_size])
    # corresponding array defining pos or neg
    train_y = list(features[:, 1][:-testing_size])

    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'pos.txt')
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)
