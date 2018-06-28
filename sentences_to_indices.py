import numpy as np

def sentences_to_indices(X, word_to_index, max_len):

    m = len(X)  # number of training examples

    X_indices = np.zeros((m, max_len))

    for i in range(m):  # loop over training examples

        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X[i]

        # Initialize j to 0
        j = 0

        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            try:
                X_indices[i, j] = word_to_index[w]
                # Increment j to j + 1
                j = j + 1
            except:
                # print("Word not found")
                temp = 1

    return X_indices