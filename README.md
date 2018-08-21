# relational-rnn-pytorch

An implementation of DeepMind's [Relational Recurrent Neural Networks](https://arxiv.org/abs/1806.01822) (Santoro et al. 2018) in PyTorch.

![](./pics/rmc.png)
![](./pics/rmc_paper_result.png)


Relational Memory Core (RMC) module is originally from [official Sonnet implementation](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/relational_memory.py). However, currently they do not provide a full language modeling benchmark code.

This repo is a port of RMC with additional comments. It features a full-fledged word language modeling benchmark vs. traditional LSTM.

It supports any arbitrary word token-based text dataset, including WikiText-2 & WikiText-103.

Both RMC & LSTM models support [adaptive softmax](https://pytorch.org/docs/stable/nn.html#adaptivelogsoftmaxwithloss) for much lower memory usage of large vocabulary dataset. RMC supports PyTorch's `DataParallel`, so you can easily experiment with a multi-GPU setup.

benchmark codes are hard-forked from [official PyTorch word-language-model example](https://github.com/pytorch/examples/tree/master/word_language_model)

# Requirements
PyTorch 0.4.1 & Python 3.6

# Examples
`python train_rmc.py --cuda ` for full training & test run of RMC with GPU.

`python train_rmc.py --cuda --adaptivesoftmax --cutoff 1000 5000 20000` if using large vocabulary dataset (like WikiText-103) to fit all the tensors in the VRAM.

`python generate_rmc.py --cuda` for generating sentences from the trained model.

`python train_rnn.py --cuda` for full training & test run of traditional RNN with GPU.

All default hyperparameters of RMC & LSTM are results from a two-week experiment using WikiText-2.

# Data Preparation
Tested with WikiText-2 and WikiText-103. WikiText-2 is bundled.

Create a subfolder inside `./data` and place word-level `train.txt`, `valid.txt`, and `test.txt` inside the subfolder.

Specify `--data=(subfolder name)` and you are good to go.

The code performs tokenization at the first training run, and the corpus is saved as `pickle`. The code will load the `pickle` file after the first run.

# WikiText-2 Benchmark Results
Both RMC & LSTM have ~11M parameters. Please refer to the training code for details on hyperparameters.

| Models        | Valid Perplexity|Test Perplexity           | Forward pass ms/batch (TITAN Xp) |  Forward pass ms/batch (TITAN V) |
|:-------------:|:-------------:|:-------------:| :-------------:| :-------------:|
| LSTM (CuDNN)      |111.31 | 105.56 | 3~3.5 | 3~4 |
| LSTM (For Loop)      |Same as CuDNN | Same as CuDNN | 13~15 | 25~30 |
| RMC      | 112.77 | 107.21      |  110~110  | 180~200|

RMC can reach a comparable performance to LSTM (with heavy hyperparameter search), but it turns out that the RMC is very slow. The multi-head self-attention at every time step may be the culprit here.
Using LSTMCell with for loop (which is more "fair" benchmark for RMC) slows down the speed by 3~5x, but it's still much faster.  

Interesting to note here is that the speed is slower in TITAN V than TITAN Xp. The reason might be that the models are relatively small and the model calls small linear operations frequently.

Maybe TITAN Xp (~1900Mhz clock speed vs. TITAN V's 1355Mhz limit) benefits from these kind of workload. Or maybe TITAN V's CUDA kernel launch latency is higher for the ops in the model.

I'm not an expert in details of CUDA. Please share your results!  

# RMC Hyperparameter Search Results
Attention parameters tend to overfit the WikiText-2. reducing the hyperparmeters for attention (key_size) can combat the overfitting.

Applying dropout at the output logit before the softmax (like the LSTM one) helped preventing the overfitting.

|embed & head size| # heads | attention MLP layers | key size | dropout at output | memory slots | test ppl|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|128|	4|	3|	128|	No|	1|	128.81 |
|128|	4|	3|	128|	No|	1|	128.81 |
|128|	8|	3|	128|	No|	1|	141.84 |
|128|	4|	3|	32|	No	|1	|123.26 |
|128|	4|	3|	32|	Yes|	1|	112.4 |
|128|	4|	3|	64|	No	|1	|124.44 |
|128|	4|	3|	64|	Yes|	1|	110.16 |
|128|	4|	2|	64|	Yes|	1|	111.67 |
|64	|4	|3	|64	|Yes	|1	|133.68 |
|64	|4	|3	|32	|Yes	|1	|135.93 |
|64	|4	|3	|64	|Yes	|4	|137.93 |
|192|	4|	3|	64|	Yes|	1|	**107.21** |
|192|	4|	3|	64|	Yes|	4|	114.85 |
|256|	4|	3|	256|	No|	1|	194.73 |
|256|	4|	3|	64|	Yes|	1|	126.39 |


# About WikiText-103
The original RMC paper presents WikiText-103 results with a larger model & batch size (6 Tesla P100, each with 64 batch size, so a total of 384. Ouch).

Using a full softmax easily blows up the VRAM. Using `--adaptivesoftmax` is highly recommended. If using `--adaptivesoftmax`, `--cutoffs` should be properly provided. Please refer to the [original API description](https://pytorch.org/docs/stable/nn.html#adaptivelogsoftmaxwithloss)

I don't have such hardware and my resource is too limited to do the experiments. Benchmark result, or any other contributions are very welcome!









