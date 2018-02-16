#Trains a neural network to classify sentences using padded or truncated sequences sent into a convolutional neural network
#Code largely followed from https://towardsdatascience.com/how-to-do-text-classification-using-tensorflow-word-embeddings-and-cnn-edae13b3e575

import os
import nltk as nl
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import lookup
from tensorflow.python.platform import gfile
import tensorflow.contrib.learn as tflearn

cwd = os.getcwd()
os.chdir('/Users/clementmanger/Desktop/Thesis/Data')

#DATA IMPORT AND TRAIN TEST SPLIT

df = pd.read_csv('ReviewsFiction.csv', sep = '|', header=0, index_col=0)
df = df[df.columns[::-1]]
msk = np.random.rand(len(df)) < 0.9

train = df[msk]
test = df[~msk]

train.to_csv('train.csv', sep = '|', header=False, index=False)
test.to_csv('test.csv', sep = '|', header=False, index=False)

length = []
for f in df['Review Text']:
    length.append(len(nl.word_tokenize(f)))
lines = df['Review Text']

#MODEL PARAMETERS

BATCH_SIZE = 32
EMBEDDING_SIZE = 10
WINDOW_SIZE = EMBEDDING_SIZE
STRIDE = int(WINDOW_SIZE/2)
TARGETS = ['True', 'False']
DEFAULTS = [['null'], ['null']]
n_classes = len(TARGETS)
MAX_DOCUMENT_LENGTH = max(length)
PADWORD = 'ZYXW'
FEATURE = 'Review Text'
LABEL = 'Fiction'

#CREATE VOCABULARY FROM LINES
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
vocab_processor.fit(lines)
# with gfile.Open('vocab.tsv', 'wb') as f:
#     f.write("{}\n".format(PADWORD))
#     for word, index in vocab_processor.vocabulary_._mapping.items():
#       f.write("{}\n".format(word))
N_WORDS = len(vocab_processor.vocabulary_)


#INPUT FUNCTION REQUIRED FOR TENSORFLOW ESTIMATOR
def train_input_fn():

    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    filename = "train.csv"
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = DEFAULTS
    col1, col2= tf.decode_csv(
        value, record_defaults=record_defaults, field_delim='|')
    label = tf.stack([col1])
    features = tf.stack([col2])
    # features = dict(zip('Review Text', col2))

    # For multiclass classification use longer 'TARGETS' attribute
    table = tf.contrib.lookup.index_table_from_tensor(
                    mapping=tf.constant(TARGETS), num_oov_buckets=0, default_value=-1)
    labels = table.lookup(label)

    return features, labels

#INPUT FUNCTION REQUIRED FOR TENSORFLOW ESTIMATOR
def eval_input_fn():

    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    filename = "test.csv"
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = DEFAULTS
    col1, col2= tf.decode_csv(
        value, record_defaults=record_defaults, field_delim='|')
    label = tf.stack([col1])
    features = tf.stack([col2])
    # features = dict(zip('Review Text', col2))

    # For multiclass classification use longer 'TARGETS' attribute
    table = tf.contrib.lookup.index_table_from_tensor(
                    mapping=tf.constant(TARGETS), num_oov_buckets=0, default_value=-1)
    labels = table.lookup(label)

    return features, labels


#ESTIMATOR SPEC
def cnn_model(features, labels, mode):

    # convert vocab to numbers
    table = lookup.index_table_from_file(
      vocabulary_file='vocab.tsv', num_oov_buckets=1, vocab_size=None, default_value=-1)

    #Looks up specific terms 'Some title'
    # numbers = table.lookup(tf.constant('Some title'.split()))
    # with tf.Session() as sess:
    #   tf.tables_initializer().run()
    #   print("{} --> {}".format(lines[0], numbers.eval()))

    #create sparse vectors, convert to dense and look vectors up in the dictionary
    # titles = tf.squeeze(features['Review Text'], [1])
    words = tf.string_split(features)
    densewords = tf.sparse_tensor_to_dense(words, default_value=PADWORD)
    numbers = table.lookup(densewords)

    #Shows dense word vectors
    # sess = tf.Session()
    #sess.run(densewords)

    #Shows vectors of words where dictionary is applied
    #table.init.run(session=sess)
    #print(numbers.eval(session=sess))

    #pads vectors out to MAX_DOCUMENT_LENGTH
    padding = tf.constant([[0,0],[0,MAX_DOCUMENT_LENGTH]])
    padded = tf.pad(numbers, padding)
    sliced = tf.slice(padded, [0,0], [-1, MAX_DOCUMENT_LENGTH])
    # sess.run(sliced)

    #create embeddings

    embeds = tf.contrib.layers.embed_sequence(sliced, vocab_size=N_WORDS, embed_dim=EMBEDDING_SIZE)
    #print('words_embed={}'.format(embeds)) # (?, 20, 10)

    #Convolutions!!!

    conv = tf.contrib.layers.conv2d(embeds, 1, WINDOW_SIZE,
                    stride=STRIDE, padding='SAME') # (?, 4, 1)
    conv = tf.nn.relu(conv) # (?, 4, 1)
    words = tf.squeeze(conv, [2]) # (?, 4)

    logits = tf.contrib.layers.fully_connected(words, n_classes, activation_fn=None)

    correctPred = tf.equal(tf.argmax(logits,1), labels)
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float64))

    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()


    if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
       loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
       train_op = tf.contrib.layers.optimize_loss(
         loss,
         tf.contrib.framework.get_global_step(),
         optimizer='Adam',
         learning_rate=0.01)
    else:
       loss = None
       train_op = None

    return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)

output_dir = '/Users/clementmanger/Desktop/tensorboard/CNN'

#Estimator instance
cnn = tf.estimator.Estimator(model_fn=cnn_model, config=tflearn.RunConfig(model_dir=output_dir))

cnn.train(input_fn=train_input_fn, steps = 5000)

cnn.evaluate(input_fn=eval_input_fn, steps = 5)
