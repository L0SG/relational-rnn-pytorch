import torch.nn as nn
import torch


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, use_cudnn_version=True,
                 use_adaptive_softmax=False, cutoffs=None):
        super(RNNModel, self).__init__()
        self.use_cudnn_version = use_cudnn_version
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if use_cudnn_version:
            if rnn_type in ['LSTM', 'GRU']:
                self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            else:
                try:
                    nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
                except KeyError:
                    raise ValueError("""An invalid option for `--model` was supplied,
                                     options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
                self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        else:
            if rnn_type in ['LSTM', 'GRU']:
                rnn_type = str(rnn_type) + 'Cell'
                rnn_modulelist = []
                for i in range(nlayers):
                    rnn_modulelist.append(getattr(nn, rnn_type)(ninp, nhid))
                    if i < nlayers - 1:
                        rnn_modulelist.append(nn.Dropout(dropout))
                self.rnn = nn.ModuleList(rnn_modulelist)
            else:
                raise ValueError("non-cudnn version of (RNNCell) is not implemented. use LSTM or GRU instead")

        if not use_adaptive_softmax:
            self.use_adaptive_softmax = use_adaptive_softmax
            self.decoder = nn.Linear(nhid, ntoken)
            # Optionally tie weights as in:
            # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
            # https://arxiv.org/abs/1608.05859
            # and
            # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
            # https://arxiv.org/abs/1611.01462
            if tie_weights:
                if nhid != ninp:
                    raise ValueError('When using the tied flag, nhid must be equal to emsize')
                self.decoder.weight = self.encoder.weight
        else:
            # simple linear layer of nhid output size. used for adaptive softmax after
            # directly applying softmax at the hidden states is a bad idea
            self.decoder_adaptive = nn.Linear(nhid, nhid)
            self.use_adaptive_softmax = use_adaptive_softmax
            self.cutoffs = cutoffs
            if tie_weights:
                print("Warning: if using adaptive softmax, tie_weights cannot be applied. Ignored.")

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if not self.use_adaptive_softmax:
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        if self.use_cudnn_version:
            output, hidden = self.rnn(emb, hidden)
        else:
            # for loop implementation with RNNCell
            layer_input = emb
            new_hidden = [[], []]
            for idx_layer in range(0, self.nlayers + 1, 2):
                output = []
                hx, cx = hidden[0][int(idx_layer / 2)], hidden[1][int(idx_layer / 2)]
                for idx_step in range(input.shape[0]):
                    hx, cx = self.rnn[idx_layer](layer_input[idx_step], (hx, cx))
                    output.append(hx)
                output = torch.stack(output)
                if idx_layer + 1 < self.nlayers:
                    output = self.rnn[idx_layer + 1](output)
                layer_input = output
                new_hidden[0].append(hx)
                new_hidden[1].append(cx)
            new_hidden[0] = torch.stack(new_hidden[0])
            new_hidden[1] = torch.stack(new_hidden[1])
            hidden = tuple(new_hidden)

        output = self.drop(output)

        if not self.use_adaptive_softmax:
            decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
            return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
        else:
            decoded = self.decoder_adaptive(output.view(output.size(0) * output.size(1), output.size(2)))
            return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM' or self.rnn_type == 'LSTMCell':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
