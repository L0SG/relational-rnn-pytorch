"""
Template to use Relational RNN module
to predict a scalar from a sequence of embeddings,
e.g. a sentence.

Input: fixed-length sequence of `num_words` words,
each represented by a `num_embedding_dims` dimensional embedding.

Output: A scalar.

Author: Jessica Yung
August 2018

Relational Memory Core implementation mostly written by Sang-gil Lee, adapted by Jessica Yung.
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from relational_rnn_general import RelationalMemory

# network params
learning_rate = 1e-3
num_epochs = 50
# dtype = torch.float

# data params
# Input = seq of `num_words` words, embedding for each word has `num_embedding_dims` dims
num_words = 10
num_embedding_dims = 5
input_size = num_embedding_dims
# Predicting a scalar
output_size = 1

num_examples = 20
test_size = 0.2
num_train = int((1 - test_size) * num_examples)
batch_size = 4

####################
# Generate data
####################

X = torch.rand((num_examples, num_words, num_embedding_dims))
# Predicting a scalar per example
y = torch.rand((num_examples, output_size))

X_train = X[:num_train]
X_test = X[num_train:]
y_train = y[:num_train]
y_test = y[num_train:]


class RMCArguments:
    def __init__(self):
        self.memslots = 1
        self.headsize = 3
        self.numheads = 4
        self.input_size = input_size  # dimensions per timestep
        self.numheads = 4
        self.numblocks = 1
        self.forgetbias = 1.
        self.inputbias = 0.
        self.attmlplayers = 3
        self.batch_size = batch_size
        self.clip = 0.1


args = RMCArguments()

device = torch.device("cpu")


####################
# Build model
####################

class RRNN(nn.Module):
    def __init__(self, batch_size):
        super(RRNN, self).__init__()
        self.memory_size_per_row = args.headsize * args.numheads
        self.relational_memory = RelationalMemory(mem_slots=args.memslots, head_size=args.headsize,
                                                  input_size=args.input_size,
                                                  num_heads=args.numheads, num_blocks=args.numblocks,
                                                  forget_bias=args.forgetbias,
                                                  input_bias=args.inputbias)
        # Map from memory to logits (categorical predictions)
        self.out = nn.Linear(self.memory_size_per_row, output_size)

    def forward(self, input, memory):
        logit, memory = self.relational_memory(input, memory)
        out = self.out(logit)

        return out, memory


model = RRNN(batch_size).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model built, total trainable params: " + str(total_params))


def get_batch(X, y, batch_num, device, batch_size=32, batch_first=True):
    if not batch_first:
        raise NotImplementedError
    start = batch_num * batch_size
    end = (batch_num + 1) * batch_size
    return X[start:end].to(device), y[start:end].to(device)


loss_fn = torch.nn.MSELoss()

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.5, patience=5)

num_batches = int(len(X_train) / batch_size)
num_test_batches = int(len(X_test) / batch_size)

memory = model.relational_memory.initial_state(args.batch_size, trainable=True).to(device)

hist = np.zeros(num_epochs)


def accuracy_score(y_pred, y_true):
    return np.array(y_pred == y_true).sum() * 1.0 / len(y_true)


####################
# Train model
####################

for t in range(num_epochs):
    epoch_loss = np.zeros(num_batches)
    # epoch_acc = np.zeros(num_batches)
    epoch_test_loss = np.zeros(num_test_batches)
    # epoch_test_acc = np.zeros(num_test_batches)
    for i in range(num_batches):
        data, targets = get_batch(X_train, y_train, i, device=device, batch_size=batch_size)
        model.zero_grad()

        # forward pass
        # replace "_" with "memory" if you want to make the RNN stateful
        y_pred, memory = model(data, memory)

        loss = loss_fn(y_pred, targets)
        loss = torch.mean(loss)
        # y_pred = torch.argmax(y_pred, dim=1)
        # acc = accuracy_score(y_pred, targets)
        epoch_loss[i] = loss
        # epoch_acc[i] = acc

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # backward pass
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        # update parameters
        optimiser.step()

    # test examples
    hist[t] = np.mean(epoch_loss).item()
    if t % 10 == 0:
        print("train: ", y_pred.squeeze().detach().cpu().numpy(), targets.squeeze().detach().cpu().numpy())
    for i in range(num_test_batches):
        with torch.no_grad():
            data, targets = get_batch(X_test, y_test, i, device=device, batch_size=batch_size)
            ytest_pred, memory = model(data, memory)

            test_loss = loss_fn(ytest_pred, targets)
            test_loss = torch.mean(test_loss)
            # ytest_pred = torch.argmax(ytest_pred, dim=1)
            # test_acc = accuracy_score(ytest_pred, targets)
            epoch_test_loss[i] = loss
            # epoch_test_acc[i] = acc

    if t % 10 == 0:
        # print(epoch_test_loss)
        # print(epoch_test_acc)
        print("Epoch {} train loss: {}".format(t, np.mean(epoch_test_loss).item()))
        print("Epoch {} test  loss: {}".format(t, np.mean(epoch_test_loss).item()))
        # print("Epoch {} train  acc: {:.2f}".format(t, np.mean(epoch_acc).item()))
        # print("Epoch {} test   acc: {:.2f}".format(t, np.mean(epoch_test_acc).item()))
        print("test: ", ytest_pred.squeeze().detach().cpu().numpy(), targets.squeeze().detach().cpu().numpy())

####################
# Plot losses
####################

plt.plot(hist, label="Training loss")
plt.legend()
plt.show()

"""
# TODO: visualise preds
plt.plot(y_pred.detach().numpy(), label="Preds")
plt.plot(y_train.detach().numpy(), label="Data")
plt.legend()
plt.show()
"""
