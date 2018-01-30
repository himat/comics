from __future__ import print_function, division
import argparse, cPickle, csv, time, sys
import h5py as h5

from preprocess.text_cloze_minibatching import *
from utils import *

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# compute accuracy over a fold
def validate(fold_name, fold_data, fold_file, val_batch_size=1024):
    print("Evaling on " + fold_name + " data")

    batches = [(x, x + val_batch_size) for x in range(0, len(fold_data[0]), val_batch_size)]
    correct = 0.
    total = 0.
    for start, end in batches:
        for batch in generate_minibatches_from_megabatch(fold_data, vdict, start, end, 
            difficulty=args.difficulty, max_unk=2, fold_dict=read_fold(fold_file, vdict), 
            shuffle_candidates=True):
            
            inputs_raw = batch[1:]
            inputs = []
            for i in inputs_raw[:-1]: # Excluding labels
                inputs.append(autograd.Variable(torch.from_numpy(i)))

            in_context_fc7, in_context_bb, in_bbmask, in_context, in_cmask, in_answer_fc7, in_answer_bb, in_answers, in_amask= inputs

            preds = model(in_context_fc7, in_answer_fc7, in_answers, in_amask)
            labels = np.argmax(batch[-1], axis=-1)
            # max_preds = np.argmax(preds, axis=1)
            _, max_preds = torch.max(preds, dim=1)
            max_preds = max_preds.view(-1,)
            max_preds = max_preds.data.cpu().numpy()

            # print("preds: " , preds.size())
            # print("max preds: ", max_preds.shape)
            # print("labels: ", labels.shape)

            for i in range(preds.size(0)):
                # print("max preds: ", max_preds[i])
                # print("labels: ", labels[i])
                if max_preds[i] == labels[i]:
                    correct += 1
                
                total += 1
    return "fold %s: got %d out of %d correct for %f accuracy" % (fold_name, correct, total, correct/total)


# In the text cloze task, we are given context panels (text and images) as well as the current panel image.
# The model needs to predict what text out of a set of candidates belongs in the textbox in the current panel.
# There is only one textbox in the current panel.

class TextOnlyNetwork(nn.Module):

    def __init__(self, d_word, d_hidden):

        super(TextOnlyNetwork, self).__init__()

        # self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        x = 3
        pass

# TODO replace all "fc7" with "images"
class ImageOnlyNetwork(nn.Module):

    def __init__(self, d_word, d_hidden):
        super(ImageOnlyNetwork, self).__init__()

        self.d_word = d_word
        self.d_hidden = d_hidden # TODO remove this once todo in forward is done
        print("d hidden: ", d_hidden)

        # TODO define constant for 4096 as number of fc7 features
        self.context_fc7_linear = nn.Linear(4096, d_hidden)
        # self.context_fc7_lstm = nn.LSTM((3, 4096), d_hidden, 1)
        self.context_fc7_lstm = nn.LSTM(input_size=d_hidden, hidden_size=d_hidden, num_layers=1)
        
        self.answer_embedding = nn.Embedding(vocab_len, d_word)

    def forward(self, context_fc7, answer_fc7, answers, a_mask):

        # print("context fc7 size: ", context_fc7.size())
        if args.cuda:
            context_fc7 = context_fc7.cuda()
            answer_fc7 = answer_fc7.cuda()
            answers = answers.cuda()
            a_mask = a_mask.cuda()

        context_length = context_fc7.size()[1]

        context_fc7 = context_fc7.view(-1, context_fc7.size(2))
        # print("context fc7 size after view: ", context_fc7.size())
        # a = context_fc7
        context_fc7_lin = self.context_fc7_linear(context_fc7)
        # b = context_fc7
        # assert(not torch.equal(a,b))
        context_fc7_lin = F.relu(context_fc7_lin)
        # print("contextfc7lin size: ", context_fc7_lin.size())

        # TODO: get the layer output size instead of using d_hidden here in case it changes in the init layer, this isn't safe
        context_fc7_lin = context_fc7_lin.view(context_length, -1, self.d_hidden)
        # print("contextfc7lin size after view: ", context_fc7_lin.size())
        context_fc7_rep_all, (h_n, c_n) = self.context_fc7_lstm(context_fc7_lin)

        context_fc7_rep = context_fc7_rep_all[0,:,:]
        # print("context_fc7_rep after lstm: ", context_fc7_rep.size())

        answers = answers.long()
        # print("answers: ", answers.size()) # Shape: 64 x 3 x 30
        # print("a type: ", type(answers.data))
        assert(answers.size(2) == 30)
        answers = answers.view(-1, answers.size(2))

        # print("answers after view: ", answers.size())
        answers_emb = self.answer_embedding(answers)
        # The "SumAverage" layer
        # answers_rep = self.answers_sumaverage([answers_emb, a_mask], compute_sum=True, num_dims=3)
        # print("answers_emb: ", answers_emb.size())
        answers_emb = answers_emb.view(-1, context_length, 30, self.d_word)
        # print("answers_emb after view: ", answers_emb.size())

        # print("a_mask: ", a_mask.size())
        a_mask = a_mask[:, :, :, None].expand_as(answers_emb)
        # print("a_mask after: ", a_mask.size())

        assert(answers_emb.dim() == 4)
        assert(answers_emb.size()[1:] == (3, 30, 256))
        assert(a_mask.dim() == 4)
        assert(a_mask.size()[1:-1] == (3, 30))

        # print(answers_emb.size())
        # print(a_mask.size())
        a_prod = answers_emb * a_mask
        # print(a_prod.size())
        # answers_rep = torch.sum(answers_emb * a_mask, dim=2, keepdim=False)
        answers_rep = torch.sum(answers_emb * a_mask, dim=2)
        answers_rep = torch.squeeze(answers_rep, dim=2)
        # print("answers_rep: ", answers_rep.size())
        assert(answers_rep.dim() == 3)
        assert(answers_rep.size()[1:] == (3, 256))

        context_fc7_rep = context_fc7_rep[:, None, :].expand_as(answers_rep) 
        # print("context_fc7_rep: ", context_fc7_rep.size())
        # Take inner product between context and the answers to produce preds
        assert(context_fc7_rep.dim() == 3)
        scores = torch.sum(context_fc7_rep * answers_rep, dim=2).squeeze(dim=2)
        preds = F.softmax(scores)
        # print("final preds: ", preds.size())
        assert(preds.dim() == 2)
        assert(preds.size(1) == (3))

        return preds

class ImageTextNetwork(nn.Module):

    def __init__(self, d_word, d_hidden):

        super(ImageTextNetwork, self).__init__()

    def forward(self, x):
        x = 4
        pass

def train():

    print("Training")

    model.train() # for putting layers into train if nec.

    epoch_loss = 0

    batch_num = 0
    num_total_batches = len(list(train_batches))
    print("Num total batches: ", num_total_batches)

    for start, end in train_batches:
        # if start > 10000:
            # break
        batch_num += 1
        print_pts = [1/4, 1/2, 3/4]
        for print_pt in print_pts:
            if batch_num == num_total_batches * print_pt:
                print(print_pt, " done")
                break

        # print("Train batches ", start, ", ", end)
        for batch in generate_minibatches_from_megabatch(train_data, vdict, start, end, max_unk=30, context_size=3, difficulty=args.difficulty, shuffle_candidates=True):
            optimizer.zero_grad()

            inputs_raw = batch[1:]
            inputs = []
            for i in inputs_raw[:-1]: # Excluding labels
                inputs.append(autograd.Variable(torch.from_numpy(i)))


            # if args.cuda:
                # for i in inputs:
                    # i.cuda()
            in_context_fc7, in_context_bb, in_bbmask, in_context, in_cmask, in_answer_fc7, in_answer_bb, in_answers, in_amask= inputs
            in_labels = inputs_raw[-1]
            # if args.cuda:
                # print("Applying cuda")
                # in_context_fc7.cuda()
                # in_answer_fc7.cuda()
                # in_answers.cuda()
                # in_amask.cuda()

            preds = model(in_context_fc7, in_answer_fc7, in_answers, in_amask)

            true_labels_one_hot = torch.from_numpy(in_labels).long()
            # Convert one hot labels to indices
            true_labels = (true_labels_one_hot == 1).nonzero()[:,1]
            true_labels = autograd.Variable(true_labels)
            if args.cuda:
                true_labels = true_labels.cuda()
            # print("true labels: ", true_labels.size())
            loss = criterion(preds, true_labels)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

    print("Loss for epoch {}: {}\n".format(epoch, epoch_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="text cloze models")
    parser.add_argument("--comics-data", default="data/comics.h5")
    parser.add_argument("--vocab", default="data/comics_vocab.p")
    parser.add_argument("--model", default="text_only", 
        help="text_only (default), image_only, image_text")
    parser.add_argument("--model-save-dir", default="saves")
    parser.add_argument("--pretrained-features", default="data/vgg_features.h5")
    parser.add_argument("--difficulty", default="easy", help="easy, hard")
    parser.add_argument("--d_word", default=256, type=int)
    parser.add_argument("--d_hidden", default=256, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--megabatch-size", default=512, type=int)
    parser.add_argument("--batch-size", default=64, type=int)

    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    BAR_WIDTH = 89 # Just for printing

    print("Loading data...")
    vdict, rvdict = cPickle.load(open(args.vocab, 'rb'))
    comics_data = h5.File(args.comics_data)
    pretrained_features = h5.File(args.pretrained_features)

    train_data = load_hdf5(comics_data["train"], pretrained_features["train"])
    dev_data = load_hdf5(comics_data["dev"], pretrained_features["dev"])
    test_data = load_hdf5(comics_data["test"], pretrained_features["test"])


    train_info_string = "Training {} model for {} text_cloze with d_word={}, d_hidden={}"
    train_info_string = colorize(train_info_string)
    print(train_info_string.format(args.model, args.difficulty, args.d_word, args.d_hidden))

    args.model_save_dir = args.model_save_dir.rstrip("/")
    model_save_file = "{}/{}_{}_{}.pt".format(args.model_save_dir, "text_cloze", args.model, args.difficulty)

    # print_hdf5_item_structure(comics_data)

    total_pages, max_panels, max_boxes, max_words = comics_data['train']['words'].shape
    vocab_len = len(vdict)

    stats_info_string = colorize("total pages: {}, max panels: {}, max boxes: {}, max_words: {}")
    print(stats_info_string.format(total_pages, max_panels, max_boxes, max_words))

    # TODO: add logging

    dev_fold = "folds/{}_{}_{}.csv".format("text_cloze", "dev", args.difficulty)
    test_fold = "folds/{}_{}_{}.csv".format("text_cloze", "test", args.difficulty)

    model_dict = {"text_only": TextOnlyNetwork,
                  "image_only": ImageOnlyNetwork,
                  "image_text": ImageTextNetwork}

    model = model_dict[args.model](args.d_word, args.d_hidden)
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = None

    train_batches = [(x, x + args.megabatch_size) for x in range(0, total_pages, args.megabatch_size)]


    # Can ctrl+C at any time to stop training early
    try:
        for epoch in range(1, args.num_epochs+1):
            epoch_start_t = time.time()
            train()
            epoch_end_t = time.time()

            # epoch_log = 'Done with epoch %d in %d seconds, loss is %f' % (epoch, epoch_end_t - epoch_start_t, epoch_loss / len(train_batches))
            epoch_log = 'Done with epoch %d in %d seconds' % (epoch, epoch_end_t - epoch_start_t)
            print(epoch_log)
            
            val_loss = validate('dev', dev_data, dev_fold)
            test_loss = validate('test', test_data, test_fold)

            print(val_loss)
            print(test_loss)

            if not best_val_loss or val_loss < best_val_loss:
                with open(model_save_file, "wb") as f:
                    torch.save(model, f)
            else:
                args.lr /= 1.0 # 4.0 # anneal the lr if no improvements in val

    except KeyboardInterrupt:
        print("-" * BAR_WIDTH)
        print("Exiting from training early")

    # Load the best saved model
    # with open(args.save, "rb") as f:
        # model = torch.load(f)

    # Can do other stuff here like run on test data
    # or save the latest model from the ctrl+C

    print("=" * BAR_WIDTH)
    print("End of training")







