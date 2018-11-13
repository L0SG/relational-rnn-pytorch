# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import datetime
import shutil
import pickle
import data
import rnn_models

# is it faster?
torch.backends.cudnn.benchmark = True

# same hyperparameter scheme as word-language-model
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.1,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=100,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true', default=True,
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--cudnn', action='store_true',
                    help='use cudnn optimized version. i.e. use RNN instead of RNNCell with for loop')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--resume', type=int, default=None,
                    help='if specified with the 1-indexed global epoch, loads the checkpoint and resumes training')

# parameters for adaptive softmax
parser.add_argument('--adaptivesoftmax', action='store_true',
                    help='use adaptive softmax during hidden state to output logits.'
                         'it uses less memory by approximating softmax of large vocabulary.')
parser.add_argument('--cutoffs', nargs="*", type=int, default=[10000, 50000, 100000],
                    help='cutoff values for adaptive softmax. list of integers.'
                         'optimal values are based on word frequencey and vocabulary size of the dataset.')

# experiment name for this run
parser.add_argument('--name', type=str, default=None,
                    help='name for this experiment. generates folder with the name if specified.')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")
###############################################################################
# Load data
###############################################################################
corpus_name = os.path.basename(os.path.normpath(args.data))
corpus_filename = './data/corpus-' + str(corpus_name) + str('.pkl')
if os.path.isfile(corpus_filename):
    print("loading pre-built " + str(corpus_name) + " corpus file...")
    loadfile = open(corpus_filename, 'rb')
    corpus = pickle.load(loadfile)
    loadfile.close()
else:
    print("building " + str(corpus_name) + " corpus...")
    corpus = data.Corpus(args.data)
    # save the corpus for later
    savefile = open(corpus_filename, 'wb')
    pickle.dump(corpus, savefile)
    savefile.close()
    print("corpus saved to pickle")


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


eval_batch_size = 32
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

# create folder for current experiments
# name: args.name + current time
# includes: entire scripts for faithful reproduction, train & test logs
folder_name = str(datetime.datetime.now())[:-7]
if args.name is not None:
    folder_name = str(args.name) + ' ' + folder_name

os.mkdir(folder_name)
for file in os.listdir(os.getcwd()):
    if file.endswith(".py"):
        shutil.copy2(file, os.path.join(os.getcwd(), folder_name))
logger_train = open(os.path.join(os.getcwd(), folder_name, 'train_log.txt'), 'w+')
logger_test = open(os.path.join(os.getcwd(), folder_name, 'test_log.txt'), 'w+')

# save args to logger
logger_train.write(str(args) + '\n')

# define saved model file location
savepath = os.path.join(os.getcwd(), folder_name)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
print("vocabulary size (ntokens): " + str(ntokens))
if args.adaptivesoftmax:
    print("Adaptive Softmax is on: the performance depends on cutoff values. check if the cutoff is properly set")
    print("Cutoffs: " + str(args.cutoffs))
    if args.cutoffs[-1] > ntokens:
        raise ValueError("the last element of cutoff list must be lower than vocab size of the dataset")
    criterion_adaptive = nn.AdaptiveLogSoftmaxWithLoss(args.nhid, ntokens, cutoffs=args.cutoffs).to(device)
else:
    criterion = nn.CrossEntropyLoss()

model = rnn_models.RNNModel(args.model, ntokens, args.emsize, args.nhid,
                            args.nlayers, args.dropout, args.tied,
                            use_cudnn_version=args.cudnn, use_adaptive_softmax=args.adaptivesoftmax,
                            cutoffs=args.cutoffs).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("model built, total trainable params: " + str(total_params))
if not args.cudnn:
    print(
        "--cudnn is set to False. the model will use RNNCell with for loop, instead of cudnn-optimzed RNN API. Expect a minor slowdown.")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

###############################################################################
# Load the model checkpoint if specified and restore the global & best epoch
###############################################################################
if args.resume is not None:
    print("--resume detected. loading checkpoint...")
global_epoch = args.resume if args.resume is not None else 0
best_epoch = args.resume if args.resume is not None else 0
if args.resume is not None:
    loadpath = os.path.join(os.getcwd(), "model_{}.pt".format(args.resume))
    if not os.path.isfile(loadpath):
        raise FileNotFoundError(
            "model_{}.pt not found. place the model checkpoint file to the current working directory.".format(
                args.resume))
    checkpoint = torch.load(loadpath)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    global_epoch = checkpoint["global_epoch"]
    best_epoch = checkpoint["best_epoch"]

print("model built, total trainable params: " + str(total_params))


###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.


def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            if not args.adaptivesoftmax:
                loss = criterion(output.view(-1, ntokens), targets)
            else:
                _, loss = criterion_adaptive(output.view(-1, args.nhid), targets)
            total_loss += len(data) * loss.item()
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    forward_elapsed_time = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)

        # synchronize cuda for a proper speed benchmark
        torch.cuda.synchronize()

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        forward_start_time = time.time()

        hidden = repackage_hidden(hidden)
        model.zero_grad()

        output, hidden = model(data, hidden)
        if not args.adaptivesoftmax:
            loss = criterion(output.view(-1, ntokens), targets)
        else:
            _, loss = criterion_adaptive(output.view(-1, args.nhid), targets)
        total_loss += loss.item()

        # synchronize cuda for a proper speed benchmark
        torch.cuda.synchronize()

        forward_elapsed = time.time() - forward_start_time
        forward_elapsed_time += forward_elapsed

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            printlog = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | forward ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                              elapsed * 1000 / args.log_interval, forward_elapsed_time * 1000 / args.log_interval,
                cur_loss, math.exp(cur_loss))
            # print and save the log
            print(printlog)
            logger_train.write(printlog + '\n')
            logger_train.flush()
            total_loss = 0.
            # reset timer
            start_time = time.time()
            forward_start_time = time.time()
            forward_elapsed_time = 0.


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    print("training started...")
    if global_epoch > args.epochs:
        raise ValueError("global_epoch is higher than args.epochs when resuming training.")
    for epoch in range(global_epoch + 1, args.epochs + 1):
        global_epoch += 1
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)

        print('-' * 89)
        testlog = '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(epoch, (
                time.time() - epoch_start_time), val_loss, math.exp(val_loss))
        print(testlog)
        logger_test.write(testlog + '\n')
        logger_test.flush()
        print('-' * 89)

        scheduler.step(val_loss)

        # Save the model if the validation loss is the best we've seen so far.
        # model_{} contains state_dict and other states, model_dump_{} contains all the dependencies for generate_rmc.py
        if not best_val_loss or val_loss < best_val_loss:
            try:
                os.remove(os.path.join(savepath, "model_{}.pt".format(best_epoch)))
                os.remove(os.path.join(savepath, "model_dump_{}.pt").format(best_epoch))
            except FileNotFoundError:
                pass
            best_epoch = global_epoch
            torch.save(model, os.path.join(savepath, "model_dump_{}.pt".format(global_epoch)))
            with open(os.path.join(savepath, "model_{}.pt".format(global_epoch)), 'wb') as f:
                optimizer_state = optimizer.state_dict()
                scheduler_state = scheduler.state_dict()
                torch.save({"state_dict": model.state_dict(),
                            "optimizer": optimizer_state,
                            "scheduler": scheduler_state,
                            "global_epoch": global_epoch,
                            "best_epoch": best_epoch}, f)
            best_val_loss = val_loss
        else:
            pass

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early: loading checkpoint from the best epoch {}...'.format(best_epoch))

# Load the best saved model.
with open(os.path.join(savepath, "model_{}.pt".format(best_epoch)), 'rb') as f:
    checkpoint = torch.load(f)
    model.load_state_dict(checkpoint["state_dict"])
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    if args.cudnn:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)

print('=' * 89)
testlog = '| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss))
print(testlog)
logger_test.write(testlog + '\n')
logger_test.flush()
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
