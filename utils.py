import numpy as np
from nltk import word_tokenize


def preprocess(input_file, batch_size, num_timesteps, genwords):
    with open(input_file) as f:

        if genwords:
            element_list = word_tokenize(f.read())
        else:
            element_list = list(f.read())
        unique_elements = set(element_list)
        element2idx = {}
        idx2element = {}
        for idx, char in enumerate(unique_elements):
            element2idx[char] = idx
            idx2element[idx] = char
        indexes = [element2idx[i] for i in element_list]
        num_batches = len(indexes) // (num_timesteps * batch_size)
        cutindex = len(indexes) % (num_batches * num_timesteps * batch_size)
        xindexes = indexes[:-cutindex]
        yindexes = indexes[1:]
        yindexes.append(element2idx['.'])
        yindexes = yindexes[:-cutindex]
        x = np.split(np.reshape(xindexes, (batch_size, -1)),
                     num_batches, axis=1)
        y = np.split(np.reshape(yindexes, (batch_size, -1)),
                     num_batches, axis=1)
        return x, y, element2idx, idx2element
