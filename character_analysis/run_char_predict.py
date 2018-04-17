################
# This file trains and then predicts on data to determine if the next word 
#   in a sentence should be a superhero/villain name or not
################
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

    cprint(f"Number of batches: {num_batches}","blue")

    data = data.reindex(np.random.permutation(data.index))

    epoch_loss = 0

    b_count = 0

    for batch in np.array_split(data, num_batches):

        print("Batch: ", b_count)
        b_count += 1

        optimizer.zero_grad()

        # text = torch.autograd.Variable(torch.from_numpy(batch[COL_TEXT].values))
        text = batch[COL_TEXT].values
        is_char_label = batch[COL_IS_CHAR].values


        # print(f"text: {text}")
        # print(f"is char label: {is_char_label}")

        curr_batch_size = text.shape[0] 
        # import pdb; pdb.set_trace()
        words = [sentence.split() for sentence in text]

        # print(words)
        unk_encoding = vdict["UNK".encode("utf-8")]

        # text_transformed_1 = []
        # for seq in words:
            # seq_transformed = []
            # for w in seq:
                # word = w.encode("utf-8")
                # if word in vdict:
                    # seq_transformed.append(vdict[word])
                # else:
                    # seq_transformed.append(unk_encoding)
            # text_transformed_1.append(seq_transformed)
                    
        text_transformed = [[vdict[w.encode("utf-8")] if w.encode("utf-8") in vdict else unk_encoding for w in seq] for seq in words]
        # text_transformed = np.array(text_transformed)
        # print(text_transformed)

        seq_lens = torch.LongTensor(list(map(len, text_transformed)))
        if args.gpuid > -1:
            seq_lens = seq_lens.cuda()


        ### convert to padded seq and then pack the padded seq
        seq_tensor = torch.autograd.Variable(torch.zeros((len(text_transformed), seq_lens.max())).long())
        if args.gpuid > -1:
            seq_tensor = seq_tensor.cuda()
        for i, (seq, seqLen) in enumerate(zip(text_transformed, seq_lens)):
            seq_tensor[i,:seqLen] = torch.LongTensor(seq)
        
        seq_lens, perm_idx = seq_lens.sort(dim=0, descending=True)
        seq_tensor = seq_tensor[perm_idx]

        ### Can only use once you use pytorch 0.4, can remove padding stuff once that happens
        # Convert to a list of Variables in sorted order using the perm_idx ordering
        # text_transformed = [torch.autograd.Variable(torch.Tensor(text_transformed[ind])) for ind in perm_idx]
        # text_packed = torch.nn.utils.rnn.pack_sequence(text_transformed)
        
        ### Packing
        # embedded_tensor = model.embedding(seq_tensor)
        # print(f"seq tens: {seq_tensor.size()}")
        # print(f"seq lens: {seq_lens.size()}")
        # print(f"embed tens: {embedded_tensor.size()}")
        # text_packed = pack_padded_sequence(embedded_tensor, seq_lens.cpu().numpy(), batch_first=True)
        # print(type(text_transformed[0]))
        # print(f"origin text: {text.shape}")
        # print(f"new text: {len(text_transformed)}")
        # print(f"packed: {len(text_packed)}")
        #####

        model.hidden = model.init_hidden(curr_batch_size)

        preds = model(seq_tensor)

        index_labels = torch.autograd.Variable(torch.LongTensor(is_char_label.astype(int)))

        # print("pred:", preds.size())
        # print("label: ", index_labels)
        # print("label s:", index_labels.size())

        loss = loss_function(preds, index_labels)
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
    args.add_argument("--batch-size", default=256)
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

    # Make sure there are no nulls
    assert(data_df[data_df.text.isnull()].empty)

    train(data_df, args, vocab_len)
 
    

