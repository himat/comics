import argparse
import os, sys
import time
import pickle

from termcolor import cprint # coloring output

import numpy as np, pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from models import CharacterPredictor
from models.character_predictor import CharacterPredictor

dirname = os.path.dirname(__file__)

EMB_DIM = 256
HIDDEN_DIM = 150
LABEL_SIZE = 2

COL_TEXT = "text"
COL_IS_CHAR = "is_char"

def train_epoch(data, des_batch_size, model, loss_function, optimizer):

    num_batches = len(data) // des_batch_size

    data = data.reindex(np.random.permutation(data.index))

    epoch_loss = 0

    for batch in np.array_split(data, num_batches):

        optimizer.zero_grad()

        # text = torch.autograd.Variable(torch.from_numpy(batch[COL_TEXT].values))
        text = batch[COL_TEXT].values
        is_char_label = batch[COL_IS_CHAR].values

        print(f"text: {text}")
        print(f"is char label: {is_char_label}")

        curr_batch_size = text.shape[0] 
        words = [sentence.split() for sentence in text]
        print(words)
        text_transformed = [[vdict[w.encode("utf-8")] for w in seq] for seq in words]
        print(text_transformed)
        # text_transformed = np.array([vdict[w] for w in words])
        print(f"origin text: {text.shape}")
        print(f"new text: {text_transformed.shape}")

        model.hidden = model.init_hidden(curr_batch_size)

        pred = model(text)

        print("pred:", pred.shape())
        print("label:", is_char_label.shape())

        loss = loss_function(is_char_label, pred)
        epoch_loss += loss.data[0]

        loss.backward()
        optimizer.step()



def train(data, args, vocab_len):

    model = CharacterPredictor(EMB_DIM, HIDDEN_DIM, vocab_len, LABEL_SIZE, args.gpuid >= 0)

    if args.gpuid >= 0:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # loss_function = nn.NLLLoss()
    loss_function = nn.CrossEntropyLoss()

    cprint("Training...", "blue")

    model.train() # Put layers into training mode

    for epoch in range(args.num_epochs):

        if epoch % args.print_epoch == 0:
            print(f"Epoch {epoch}")
    
        epoch_loss = train_epoch(data, args.batch_size, model, loss_function, optimizer)

        if epoch % args.print_epoch == 0:
            print(f"Epoch loss: {epoch_loss}")


    print("Done training")

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--lr", default=1e-4)
    args.add_argument("--batch-size", default=4)
    args.add_argument("--num-epochs", default=10)

    args.add_argument("--data-file", default="data/character_identity_cloze.csv")
    args.add_argument("--vocab", default="../data/comics_vocab.p")

    args.add_argument("--print-epoch", default=1)

    args.add_argument("--gpuid", default=-1, type=int)
    return args.parse_args()

if __name__=="__main__":
    args = parse_args()

    cprint("change batch size", "red")

    if torch.cuda.is_available() and args.gpuid < 0:
        cprint("WARNING: You have a CUDA device, so you should run with --gpuid 0", "red")

    if args.gpuid >= 0:
        torch.cuda.set_device(args.gpuid)
        using_gpu_string = "Using GPU {}"
        print(colorize("Using GPU {}").format(torch.cuda.current_device()))

    cprint("Loading data...", "cyan")

    if sys.version_info[0] < 3:
        raise Exception("Using Python 2, use Python 3+ instead")
        # vdict, rvdict = pickle.load(open(args.vocab, 'rb'))
    
    vdict, rvdict = pickle.load(open(args.vocab, 'rb'), encoding="bytes")
    vocab_len = len(vdict)

    character_cloze_data_file = os.path.join(dirname, args.data_file)
    data_df = pd.read_csv(character_cloze_data_file)

    data_df.info()
    print(data_df.head())

    train(data_df, args, vocab_len)
 
    

