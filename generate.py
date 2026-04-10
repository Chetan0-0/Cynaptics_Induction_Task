import torch
import tiktoken
from model import GPT

# hyperparameter
block_size = 128
n_embd = 128
n_head = 4
n_layer = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#loading tokenizer dictionary
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

print("initializing empty model block")
model = GPT(vocab_size, n_embd, n_head, n_layer, block_size)

print("loading trained weights")
model.load_state_dict(torch.load('gpt2_shakespeare.pt', map_location=device, weights_only=True))
model.to(device)
model.eval()

print("\n generating shakespeare \n")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_indices = model.generate(context, max_new_tokens=300)
generated_text = enc.decode(generated_indices[0].tolist())

print(generated_text)
