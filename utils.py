import numpy as np


def preprocess(input_file, batch_size, num_timesteps):
    with open(input_file) as f:
        charlist = list(f.read())
        unique_chars = set(charlist)
        char2idx = {}
        idx2char = {}
        for idx, char in enumerate(unique_chars):
            char2idx[char] = idx
            idx2char[idx] = char
        indexes = [char2idx[i] for i in charlist]
        num_batches = len(indexes) / (num_timesteps * batch_size)
        cutindex = len(indexes) % (num_batches * num_timesteps * batch_size)
        xindexes = indexes[:-cutindex]
        yindexes = indexes[1:]
        yindexes.append(char2idx['.'])
        yindexes = yindexes[:-cutindex]
        x = np.split(np.reshape(xindexes, (batch_size, -1)),
                     num_batches, axis=1)
        y = np.split(np.reshape(yindexes, (batch_size, -1)),
                     num_batches, axis=1)
        return x, y, char2idx, idx2char
