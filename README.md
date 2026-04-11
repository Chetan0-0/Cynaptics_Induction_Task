# Cynaptics_Induction_Task
# Building a mini GPT-2 from scratch

Hello this is my submission for Cynaptics Task 1. I built a decoder only Transformer model from scratch using PyTorch. The architecture is based on GPT-2, and the goal was to train it to generate text by predicting the next token.

# Project Overview

The task instructions specifically asked for a sub-word level causal language model. Instead of building a basic character-level model (which often throws OOV/Unknown errors on new words), I decided to use OpenAI's `tiktoken` library to implement the actual GPT-2 BPE tokenizer. This gives the model a proper vocabulary of 50,257 sub-words to work with.

# The Dataset

For training, we were provided with the Tiny Shakespeare dataset. 
* **Source Link:** (https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
* **Processing:** My `train.py` script downloads this raw text file and translates the entire play into integer tensors using the `tiktoken` dictionary.
* **Splitting:** I sliced the dataset into a 90% training block and a 10% validation block so I could actually track the validation loss and make sure the model wasn't just memorizing the script.

# Things I Learned

Since I built the architecture from the ground up, I spent a lot of time studying the underlying topics. Some of the core concepts I learned and implemented for this task include:
* **Multi-Head Attention:** Understanding how to project tokens into Queries, Keys and Values so they can gather context from each other and how splitting them into multiple heads lets the model track different relationships.
* **Causal Masking:** Using lower-triangular matrices (`torch.tril`) to mask out future tokens with `-inf` so the model can't look ahead in data during training.
* **FeedForward Networks (FFN):** Adding the computation layers after the attention mechanism to give the model time to think about the context it just got.
* **Architecture & Hyperparameters:** Learning exactly how `block_size`, `batch_size`, and `n_embd` dictate the `(B, T, C)` tensor shapes as data flows through the residual connections.

# Hyperparameters
* `vocab_size` = 50,257 
* `block_size` = 128 (Context window)
* `n_embd` = 128 (Embedding dimension)
* `n_head` = 4 (Attention heads)
* `n_layer` = 4 (Transformer blocks)
* `batch_size` = 32

# Repository Structure
* `model.py`: The core math and architecture (Blocks, Attention, FFN, and the main GPT class).
* `train.py`: Downloads the data, sets up the tokenizer, runs the training loop and saves the trained `.pt` weights.
* `generate.py`: Loads the trained `.pt` dictionary into an empty model chassis to auto-generate new Shakespeare text.

# Sample Output
These are the screen shots of the output from task 1 
<img width="305" height="81" alt="Screenshot 2026-04-11 201619" src="https://github.com/user-attachments/assets/49e85d03-5b42-4d03-a17e-79b0651e9027" />
<img width="602" height="696" alt="Screenshot 2026-04-11 201638" src="https://github.com/user-attachments/assets/ddb22a88-dd1d-4cd2-b7af-84047e116f71" />
