import os
import requests
import torch
import tiktoken
from model import GPT

# hyperparams
batch_size = 32
block_size = 128
n_embd = 128
n_head = 4
n_layer = 4
max_iters = 1000
eval_interval = 250
eval_iters = 100
learning_rate = 1e-3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"currently running on: {device}")

#dataset loading
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
file_path = "input.txt"

if not os.path.exists(file_path):
    print("downloading the dataset...")
    response = requests.get(url)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print("download done!")

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

#tokenizing using openais subword bpe
enc = tiktoken.get_encoding("gpt2")
data = torch.tensor(enc.encode(text), dtype=torch.long)
vocab_size = enc.n_vocab

#train valuation data spliting
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    split_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(split_data) - block_size, (batch_size,))
    x = torch.stack([split_data[i : i + block_size] for i in ix])
    y = torch.stack([split_data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

model = GPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#testing loop with gradients turned off so it doesn't memorize the val set
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print("starting training")
for iter in range(max_iters):

    #check validation loss to prevent overfitting
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}- train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#saving the model dictionary
torch.save(model.state_dict(), 'gpt2_shakespeare.pt')
print("model saved to 'gpt2_shakespeare.pt'")
