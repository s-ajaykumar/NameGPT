import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# Bigram Model Building
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, ix, targets = None):
        logits = self.token_embedding_table(ix)
        
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, ix, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(ix)
            logits = logits[:, -1, :]
            p_dis = F.softmax(logits, dim = -1)
            next_ix = torch.multinomial(p_dis, num_samples = 1)
            ix = torch.cat((ix, next_ix), dim = -1)
        return ix
    
 
# Hyperparameters
max_iters = 3000
eval_iters = 200
eval_interval = 300
lr = 1e-2
batch_size = 32
block_size = 8

   
# EDA
text = open("Dataset/shakesphere.txt").read()
vocab = sorted(set(text))
vocab_size = len(vocab)
print(f"Vocabulary : {vocab}")
print() 
print(f"Vocab size : {vocab_size}")
print() 


# Dataset Preprocessing
stoi = {ch:i+1 for i, ch in enumerate(vocab[1:])}
itos = {i+1:ch for i, ch in enumerate(vocab[1:])}
stoi['\n'] = 0
itos[0] = '\n'
encode = lambda text: [stoi[ch] for ch in text]
decode = lambda ix: ''.join([itos[i] for i in ix])
text = torch.tensor(encode(text), dtype = torch.long)
n1 = int(0.9 * len(text))
train_data = text[:n1]
val_data = text[n1:]
def get_batch(mode):
    data = train_data if mode == "train" else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i : i+block_size] for i in ix])
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix])
    return x, y
 
    
# Model Initialization
torch.manual_seed(1337)
model = BigramLanguageModel()
print(f"Parameter count : {sum([p.nelement() for p in model.parameters()])}")
print() 
print(f"Model response(Before training)\n{decode(model.generate(torch.zeros((1, 1), dtype = torch.long), 500).tolist()[0])}")
print() 
 
 
#Train + Evaluation
optimizer = torch.optim.AdamW(model.parameters(), lr)
for i in range(max_iters):
    if i%eval_interval == 0:
        losses = estimate_loss()
        print(f"step {i}   : train_loss : {losses["train"]:2f}    val_loss : {losses["val"]:2f}")
    x, y = get_batch("train")
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
print()


#Inference
print(f"Model response(After training)\n{decode(model.generate(torch.zeros((1, 1), dtype = torch.long), 500).tolist()[0])}")