import os, sys
import numpy as np
import torch

dirname = os.path.dirname(__file__)
raw_glove_vec_file = os.path.join(dirname, "../data/glove/glove.6B.{EMB_DIM}d.txt")

# Creates a dict mapping glove words to vectors
def load_glove_embeddings(emb_dim, word2idx):
    assert emb_dim in [50, 100, 200, 300]
    glove_vec_file = raw_glove_vec_file.format(EMB_DIM=emb_dim)
    print("formatted glove file: ", glove_vec_file)

    embeddings = np.zeros((len(word2idx), emb_dim))

    with open(glove_vec_file) as f:
        for line in f.readlines():
            values = line.split()
            word = values[0]
            index = word2idx.get(word.encode("utf-8")) # TODO: fix dict so encoding isn't nec
            if index:
                vector = np.array(values[1:], dtype="float32")
                embeddings[index] = vector

        return torch.from_numpy(embeddings).float()


