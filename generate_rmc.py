###############################################################################

# This file generates new sentences sampled from the language model

###############################################################################

import argparse

import torch
import pickle
import data
import os

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

if args.checkpoint is None:
    raise ValueError("--checkpoint not provided. specify model_dump_(epoch).pt")

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

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

ntokens = len(corpus.dictionary)
memory = model.module.initial_state(1, trainable=False).to(device)

input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

with open(args.outf, 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.words):
            output, _, memory = model(input, memory, None, require_logits=True)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)
            word = corpus.dictionary.idx2word[word_idx]

            outf.write(word + ('\n' if i % 20 == 19 else ' '))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
