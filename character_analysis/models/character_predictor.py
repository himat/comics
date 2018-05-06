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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CharacterPredictor(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, pretrained_emb=None):
        super(CharacterPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu

        assert pretrained_emb.shape == (vocab_size, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_emb is not None:
            self.embedding.weight = nn.Parameter(pretrained_emb)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=False)
        self.num_dirs = (2 if self.lstm.bidirectional else 1)
        assert self.num_dirs == 1

        print(f"numdirs: {self.num_dirs}")

        self.hidden_to_label = nn.Linear(self.hidden_dim*self.num_dirs, label_size)
        self.hidden = None #self.init_hidden()


    def init_hidden(self, batch_size):
        # (h, c)
        # (num_dirs_times_num_layers, batch_size, hidden_dim)
        h = torch.zeros(self.num_dirs, batch_size, self.hidden_dim)
        c = torch.zeros(self.num_dirs, batch_size, self.hidden_dim)
 
        if self.use_gpu:
            h = h.cuda()
            c = c.cuda()

        return (Variable(h), Variable(c))
        

    def forward(self, sentence):
        # print("sent: ", sentence.size())
        x = self.embedding(sentence)
        # x = sentence

        # print("x: ", x.size())
        x = x.transpose(0,1) # Since not using batch first
        # print("x: ", x.size())
        # print("hidden: ", self.hidden[0].size())

        lstm_out, self.hidden = self.lstm(x, self.hidden)

        # lstm_out = pad_packed_sequence(lstm_out)
        # print(lstm_out[-1])
        # last_output = torch.autograd.Variable(torch.FloatTensor(lstm_out[-1]))
        # print("last out: ", last_output.size())

        y = self.hidden_to_label(lstm_out[-1])
        log_probs = F.log_softmax(y, dim=1)

        return log_probs


