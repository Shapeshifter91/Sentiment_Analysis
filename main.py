from utils import *
from sentences_to_indices import *
from analyzer import *
from tokenize_line import *
from keras import optimizers
import h5py
from keras.models import load_model

from sklearn.model_selection import train_test_split


# Read Data
X = []
Y = []
with open('data/yelp_labelled.txt', 'r') as f:
    for line in f:
        lines_split = tokenize_line(line)
        X.append(lines_split[0:-1])
        Y.append(lines_split[-1])
with open('data/imdb_labelled.txt', 'r') as f:
    for line in f:
        lines_split = tokenize_line(line)
        X.append(lines_split[0:-1])
        Y.append(lines_split[-1])
with open('data/amazon_cells_labelled.txt', 'r') as f:
    for line in f:
        lines_split = tokenize_line(line)
        X.append(lines_split[0:-1])
        Y.append(lines_split[-1])

maxLen = len(max(X, key=len))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=101)

Y_train = np.array([int(temp) for temp in Y_train])
Y_test = np.array([int(temp) for temp in Y_test])

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')

embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

model = SentimentAnalyzer((maxLen,), word_to_vec_map, word_to_index)
model.summary()

opti = optimizers.adam(0.0001)
model.compile(loss='binary_crossentropy', optimizer=opti, metrics=['accuracy'])

X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
# Y_train_oh = Y_train
Y_train_oh = convert_to_one_hot(Y_train, C = 2)

model.fit(X_train_indices, Y_train_oh, epochs = 45, batch_size = 32, shuffle=True)

X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
# Y_test_oh = Y_test
Y_test_oh = convert_to_one_hot(Y_test, C = 2)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print("Test accuracy = ", acc)


model.save('my_model.h5')


y_predict = model.predict(X_test_indices)
space_s = " "
for i in range(len(X_test)):
    print('{} {}'.format(space_s.join(X_test[i]),-y_predict[i][0]+y_predict[i][1]))







