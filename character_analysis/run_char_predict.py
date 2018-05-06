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
sys.path.append('..')
from utils import glove

dirname = os.path.dirname(__file__)

# TODO: try 300
EMB_DIM = 200#256
HIDDEN_DIM = 150
LABEL_SIZE = 2

COL_TEXT = "text" # x
COL_IS_CHAR = "is_char" # y
COL_NEXT_WORD = "next_word"
COL_CHAR_TYPE = "char_type"
COL_POST_TEXT = "post_text"

def process_input_batch(batch, args):
    text = batch[COL_TEXT].values
    is_char_labels = batch[COL_IS_CHAR].values

    words = [sentence.split() for sentence in text]
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
    # print(len(words))
    # print(len(text_transformed))
    # print("Len of all sentences")
    # for l in text_transformed:
        # print(len(l))
    # print("---")

    do_padding = False

    if not do_padding:
        assert args.batch_size == 1, "batch size should be 1 if not using padding"
        seq_tensor = torch.LongTensor(text_transformed)
        seq_tensor = torch.autograd.Variable(seq_tensor)

    else:
        seq_lens = torch.LongTensor(list(map(len, text_transformed)))
        # if args.gpuid > -1:
            # seq_lens = seq_lens.cuda()
        _, perm_idx = seq_lens.sort(dim=0, descending=True)

        # Pad and then pack
        seq_tensor = torch.autograd.Variable(torch.zeros((len(text_transformed), seq_lens.max())).long())
        # if args.gpuid > -1:
            # seq_tensor = seq_tensor.cuda()
        for i, (seq, seqLen) in enumerate(zip(text_transformed, seq_lens)):
            seq_tensor[i, :seqLen] = torch.LongTensor(seq)

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

    # labels
    index_labels = torch.autograd.Variable(torch.LongTensor(is_char_labels.astype(int)))
    # if args.gpuid > -1:
        # index_labels = index_labels.cuda()

    if args.gpuid > -1:
        seq_tensor = seq_tensor.cuda()

    return seq_tensor, index_labels

def train_epoch(data, args, model, loss_function, optimizer):

    num_batches = len(data) // args.batch_size 

    print(f"Number of batches: {num_batches}")

    data = data.reindex(np.random.permutation(data.index))

    # epoch_loss = 0
    epoch_avg_loss = 0

    b_count = 0

    print_mod = max(num_batches // 4, 1)
    
    for batch in np.array_split(data, num_batches):

        if b_count % print_mod == 0:
            print(f"Batch: {b_count}/{num_batches}")
        b_count += 1

        # optimizer.zero_grad()
        model.zero_grad()

        curr_batch_size = batch.shape[0] 
        seq_tensor, index_labels = process_input_batch(batch, args)

        model.hidden = model.init_hidden(curr_batch_size)

        preds = model(seq_tensor)

        if args.gpuid > -1:
            preds = preds.cpu()

        loss = loss_function(preds, index_labels)
        epoch_avg_loss += loss.data[0]

        loss.backward()
        optimizer.step()

    epoch_avg_loss /= data.shape[0]
    return epoch_avg_loss


def train(train_data, test_data, args, vocab_len):

    pretrained_emb_vec = None
    if args.pretrained_emb:
        cprint("Using pretrained embeddings", "cyan")
        pretrained_emb_vec = glove.load_glove_embeddings(EMB_DIM, vdict)

    model = CharacterPredictor(EMB_DIM, HIDDEN_DIM, vocab_len, LABEL_SIZE, args.gpuid >= 0, pretrained_emb=pretrained_emb_vec)

    if args.gpuid >= 0:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.NLLLoss() # Add a log_softmax layer to model if using this loss
    # loss_function = nn.CrossEntropyLoss()

    cprint("Training...", "blue")

    model.train() # Put layers into training mode

    for epoch in range(args.num_epochs):

        if epoch % args.print_epoch == 0:
            print(f"\nEpoch {epoch}")
    
        epoch_avg_loss = train_epoch(train_data, args, model, loss_function, optimizer)

        if epoch % args.print_epoch == 0:
            print(f"Epoch avg loss: {epoch_avg_loss}")

        # TODO: no need to  go through all data again, just calculate stats in train_epoch
        validate(model, train_data, args, data_name="train")
        validate(model, test_data, args, data_name="test")

    print("Done training")

    return model


def validate(model, data, args, data_name=""):

    cprint(f"Validation mode {data_name}", "blue")

    num_batches = len(data) // args.batch_size

    num_total = 0
    num_correct = 0

    num_true_chars = 0
    num_true_nonchars = 0
    num_pred_chars = 0
    num_pred_nonchars = 0

    num_correct_pred_chars = 0
    num_correct_pred_nonchars = 0

    for batch in np.array_split(data, num_batches):

        curr_batch_size = batch.shape[0]
    
        seq_tensor, index_labels_var = process_input_batch(batch, args)
        index_labels = index_labels_var.data.cpu()

        model.hidden = model.init_hidden(curr_batch_size)

        preds_probs = model(seq_tensor)
        _, preds_index = torch.max(preds_probs.data, dim=1)

        preds_index = preds_index.cpu()
        
        num_total += curr_batch_size
        num_correct += (preds_index == index_labels).sum()
       
        true_chars = (index_labels == 1)
        true_nonchars = (index_labels == 0)
        pred_chars = (preds_index == 1)
        pred_nonchars = (preds_index == 0)

        num_true_chars += true_chars.sum()
        num_true_nonchars += true_nonchars.sum()
        num_pred_chars += pred_chars.sum()
        num_pred_nonchars += pred_nonchars.sum()
       
        # Find where both vectors have 1's
        num_correct_pred_chars += (true_chars * pred_chars).sum()
        num_correct_pred_nonchars += (true_nonchars * pred_nonchars).sum()

        # print(batch)
        # print("true labels: ", index_labels)
        # print("pred labels: ", preds_index)
        # print(f"true chars: {true_chars}")
        # print(f"true nonchars: {true_nonchars}")
        # print(f"pred chars: {pred_chars}")
        # print(f"pred nonchars: {pred_nonchars}")
        
        # break


    accuracy = num_correct / num_total
    print(f"Accuracy {data_name}: {num_correct}/{num_total} ({accuracy*100}%)")

    print(f"True chars: {num_true_chars} | True non-chars: {num_true_nonchars}")
    print(f"Pred chars: {num_pred_chars} | Pred non-chars: {num_pred_nonchars}")
    
    print(f"Num correctly predicted chars: {num_correct_pred_chars}") 
    print(f"Num correctly predicted non-chars: {num_correct_pred_nonchars}") 

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--batch-size", type=int, default=256)
    args.add_argument("--num-epochs", type=int, default=10)

    args.add_argument("--train-data-file", default="data/character_identity_cloze_train.csv")
    args.add_argument("--test-data-file", default="data/character_identity_cloze_test.csv")
    args.add_argument("--vocab", default="../data/comics_vocab.p")

    args.add_argument("--pretrained-emb", action="store_true", help="Use pretained word embeddings")

    args.add_argument("--print-epoch", type=int, default=1)
    args.add_argument("--data-limit", type=int, default=None, 
        help="Pass in the number of rows you want the model to train and test for (useful for testing code)")

    args.add_argument("--gpuid", default=-1, type=int)
    return args.parse_args()

if __name__=="__main__":
    args = parse_args()
    print(args)

    if torch.cuda.is_available() and args.gpuid < 0:
        cprint("WARNING: You have a CUDA device, so you should run with --gpuid 0", "red")

    if args.gpuid >= 0:
        torch.cuda.set_device(args.gpuid)
        using_gpu_string = "Using GPU {}"
        cprint(f"Using GPU #{torch.cuda.current_device()}", "yellow")

    cprint("Loading data...", "cyan")

    if sys.version_info[0] < 3:
        raise Exception("Using Python 2, use Python 3+ instead")
        # vdict, rvdict = pickle.load(open(args.vocab, 'rb'))
    
    vdict, rvdict = pickle.load(open(args.vocab, 'rb'), encoding="bytes")
    vocab_len = len(vdict)

    col_dtypes = {COL_TEXT: str, COL_NEXT_WORD: str, COL_IS_CHAR: bool, COL_CHAR_TYPE: str}

    character_cloze_train_file = os.path.join(dirname, args.train_data_file)
    # Not using default na values since it turns the "nan" string which appears 
    #   in the corpus as a word into an actual NaN 
    train_data_df = pd.read_csv(character_cloze_train_file, dtype=col_dtypes, keep_default_na=False, na_values={COL_NEXT_WORD: ""})
    
    character_cloze_test_file = os.path.join(dirname, args.test_data_file)
    test_data_df = pd.read_csv(character_cloze_test_file, dtype=col_dtypes, keep_default_na=False, na_values={COL_NEXT_WORD: ""})

    # Get data up to the limit
    # Note that this uses the unshuffled data, so all data of a class will be grouped together
    if args.data_limit:
        if args.data_limit < args.batch_size:
            raise ValueError("Amount of used data cannot be smaller than batch size")

        train_data_df = train_data_df[:args.data_limit]
        test_data_df = test_data_df[:args.data_limit]

    train_data_df = train_data_df.sample(frac=1).reset_index(drop=True)
    test_data_df = test_data_df.sample(frac=1).reset_index(drop=True)
    
    cprint("Train data", "blue")
    train_data_df.info()
    print(train_data_df.head())
    cprint("Test data", "blue")
    test_data_df.info()
    print(test_data_df.head())

    # Make sure there are no nulls anywhere
    # It's fine for post text to have nulls (empty strings)
    assert(not train_data_df.drop([COL_POST_TEXT], axis=1).isnull().values.any())
    assert(not test_data_df.drop([COL_POST_TEXT], axis=1).isnull().values.any())

    model = train(train_data_df, test_data_df, args, vocab_len)

    validate(model, test_data_df, args, data_name="test")
 
    

