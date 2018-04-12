from __future__ import print_function, division
import argparse, csv, time, sys
import pickle
import h5py as h5

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CharacterPredictor(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu):
        super(CharacterPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.num_dirs = (2 if self.lstm.bidirectional else 1)

        print(f"numdirs: {self.num_dirs}")

        self.hidden_to_label = nn.Linear(self.hidden_dim*self.num_dirs, label_size)
        self.hidden = None #self.init_hidden()


    def init_hidden(self, batch_size):
        # (h, c)
        # (num_dirs_times_num_layers, batch_size, hidden_dim)
        h = torch.zeros(self.num_dirs, batch_size, self.hidden_dim)
        c = torch.zeros(self.num_dirs, batch_size, self.hidden_dim)
 
        if self.use_gpu:
            return (Variable(h.cuda()), Variable(c.cuda()))
        else:
            return (Variable(h), Variable(c))
        

    def forward(self, sentence):
        x = self.embedding(sentence)
        print(f"x shape: {x.size()}")
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden_to_label(lstm_out[-1])
        log_probs = F.log_softmax(y)

        return log_probs


