import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

n_nodes_hl1 = 500
n_nodes_hl2 = 500

n_classes = 2

batch_size = 512
total_batches = int(1600000/batch_size)
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum': n_nodes_hl1,
                  'weight': tf.Variable(tf.random_normal([3010, n_nodes_hl1])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum': n_nodes_hl2,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum': None,
                'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias': tf.Variable(tf.random_normal([n_classes]))}


def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    output = tf.matmul(l2, output_layer['weight']) + output_layer['bias']
    return output


saver = tf.train.Saver()
tf_log = 'tf.log'


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 16
    config.inter_op_parallelism_threads = 2
    with tf.Session(config=config) as sess:
        sess.run(tf.initialize_all_variables())

        try:
            epoch = int(open(tf_log, 'r').read().split('\n')[-2])+1
            print('STARTING:', epoch)
        except:
            epoch = 1

        while epoch <= hm_epochs:
            if epoch != 1:
                saver.restore(sess, "./model.ckpt")
            epoch_loss = 1
            with open('lexicon.pickle', 'rb') as f:
                lexicon = pickle.load(f)
            with open('train_set_shuffled.csv', buffering=2000000, encoding='latin-1') as f:
                batch_x = []
                batch_y = []
                batches_run = 0
                for line in f:
                    label = line.split(':::')[0]
                    tweet = line.split(':::')[1]
                    current_words = word_tokenize(tweet.lower())
                    current_words = [lemmatizer.lemmatize(i) for i in current_words]

                    features = np.zeros(len(lexicon))

                    for word in current_words:
                        if word.lower() in lexicon:
                            index_value = lexicon.index(word.lower())
                            # OR DO +=1, test both
                            features[index_value] += 1
                    line_x = list(features)
                    line_y = eval(label)
                    batch_x.append(line_x)
                    batch_y.append(line_y)
                    if len(batch_x) >= batch_size:
                        _, c = sess.run([optimizer, cost], feed_dict={x: np.array(batch_x),
                                                                      y: np.array(batch_y)})
                        epoch_loss += c
                        batch_x = []
                        batch_y = []
                        batches_run += 1
                        print('Batch run:', batches_run, '/', total_batches, '| Epoch:', epoch, '| Batch Loss:', c)

            saver.save(sess, "./model.ckpt")
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            with open(tf_log, 'a') as f:
                f.write(str(epoch)+'\n')
            epoch += 1


# train_neural_network(x)


def test_neural_network():
    prediction = neural_network_model(x)
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 16
    config.inter_op_parallelism_threads = 2
    with tf.Session(config=config) as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, "./model.ckpt")

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        feature_sets = []
        labels = []
        counter = 0
        with open('processed-test-set.csv', buffering=20000) as f:
            for line in f:
                try:
                    features = list(eval(line.split('::')[0]))
                    label = list(eval(line.split('::')[1]))
                    feature_sets.append(features)
                    labels.append(label)
                    counter += 1
                except:
                    pass
        print('Tested', counter, 'samples.')
        test_x = np.array(feature_sets)
        test_y = np.array(labels)
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))


test_neural_network()


def use_neural_network(input_data):
    prediction = neural_network_model(x)
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 16
    config.inter_op_parallelism_threads = 2
    with open('lexicon.pickle', 'rb') as f:
        lexicon = pickle.load(f)

    with tf.Session(config=config) as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, "./model.ckpt")
        current_words = word_tokenize(input_data.lower())
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))

        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                # OR DO +=1, test both
                features[index_value] += 1

        features = np.array(list(features))
        # pos: [1,0] , argmax: 0
        # neg: [0,1] , argmax: 1
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}), 1)))
        print(prediction.eval(feed_dict={x:[features]}))
        if result[0] == 0:
            print('Positive:', input_data)
        elif result[0] == 1:
            print('Negative:', input_data)


use_neural_network("He's an idiot and a jerk.")
use_neural_network("This was the best like amazing store i've ever seen.")
use_neural_network("I'm itchy and miserable!")
