import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from relational_rnn_general import RelationalMemory

# network params
learning_rate = 1e-3
num_epochs = 50
dtype = torch.float

# data params
num_vectors = 4
num_dims = 2
num_examples = 20
test_size = 0.2
num_train = int((1-test_size) * num_examples)
batch_size = 4

####################
# Generate data
####################

# For each example
X = np.zeros((num_examples, num_vectors*(num_dims+1)+2))
y = np.zeros(num_examples)

for i in range(num_examples):
    n = np.random.choice(num_vectors, 1) # nth farthest from target vector
    labels = np.random.choice(num_vectors,num_vectors,replace=False)
    m_index = np.random.choice(num_vectors, 1) # m comes after the m_index-th vector
    m = labels[m_index]

    vectors = np.random.rand(num_vectors, num_dims)*2 - 1
    target_vector = vectors[m_index]
    dist_from_target = np.linalg.norm(vectors - target_vector, axis=1)
    X_single = np.zeros((num_vectors, num_dims + 1))
    X_single[:, :-1] = vectors
    X_single[:, -1] = labels
    X_single = np.concatenate((X_single.reshape(-1), np.array([m,n]).reshape(-1)))
    y_single = labels[np.argsort(dist_from_target)[-n]]

    X[i,:] = X_single
    y[i] = y_single

seq_len = num_vectors * (num_dims+1) + 5

X = torch.Tensor(X)
y = torch.Tensor(y)

X_train = X[:num_train]
X_test = X[num_train:]
y_train = y[:num_train]
y_test = y[num_train:]


class RMCArguments:
    def __init__(self):
        self.memslots = 1
        self.headsize = 3
        self.numheads = 4
        self.input_size = 1 # or input_size
        self.numheads = 4
        self.numblocks = 1
        self.forgetbias = 1.
        self.inputbias = 0.
        self.attmlplayers = 3
        self.cutoffs = [10000, 50000, 100000]
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
        self.relational_memory = RelationalMemory(mem_slots=args.memslots, head_size=args.headsize, input_size=args.input_size,
                         num_heads=args.numheads, num_blocks=args.numblocks, forget_bias=args.forgetbias,
                         input_bias=args.inputbias,
                         attention_mlp_layers=args.attmlplayers, use_adaptive_softmax=args.adaptivesoftmax,
                         cutoffs=args.cutoffs).to(device)
        self.relational_memory = nn.DataParallel(self.relational_memory)
        self.out = nn.Linear(self.memory_size_per_row, batch_size)

    def forward(self, input, memory):
        memory = self.relational_memory(input, memory)[-1]
        out = self.out(memory)

        return out

model = RRNN(batch_size)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model built, total trainable params: " + str(total_params))

# # TODO: should forget & input bias be trainable? sonnet is not i think
# model.forget_bias.requires_grad = False
# model.input_bias.requires_grad = False

def get_batch(X, y, batch_num, batch_size=32, batch_first=True):
    if not batch_first:
        raise NotImplementedError
    start = batch_num*batch_size
    end = (batch_num+1)*batch_size
    return X[start:end], y[start:end]

loss_fn = torch.nn.MSELoss(size_average=False)

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.5, patience=5)

num_batches = int(len(X_train) / batch_size)

memory = model.relational_memory.module.initial_state(args.batch_size, trainable=True).to(device)

hist = np.zeros(num_epochs)

####################
# Train model
####################

for t in range(num_epochs):
    for i in range(num_batches):
        data, targets = get_batch(X_train, y_train, i, batch_size=batch_size)
        model.zero_grad()
        # model.hidden = model.init_hidden()
        # forward pass
        y_pred = model(data, memory)

        loss = loss_fn(y_pred, targets)
        loss = torch.mean(loss)
        hist[t] = loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # backward pass
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        # update parameters
        optimiser.step()
    if t % 10 == 0:
        print("Epoch ", t, "MSE: ", loss.item())

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

