
def preprocess(input_file):
    with open(input_file) as f:
        charlist = list(f.read())
        unique_chars = set(charlist)
        char2idx = {}
        idx2char = {}
        for idx, char in enumerate(unique_chars):
            char2idx[char] = idx
            idx2char[idx] = char
        idxlist = [char2idx[i] for i in charlist]
        return idxlist, charlist, char2idx, idx2char
