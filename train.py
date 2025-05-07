import torch
import random
import torch.nn as nn
import torch.nn.functional as F


# Hyperparmeters
batch_size = 4
block_size = 23
n_embd = 384
n_heads = 6
n_blocks = 6
dropout_ratio = 0.2
lr = 3e-4
max_iters = 5001
eval_interval = 500
eval_iters = 200
pad_token = 57
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)


# EDA
text = open("Dataset/names.txt").read()
vocab = sorted(set(text))
vocab_size = len(vocab) + 1     # 1 refers to pad token
print("-"*80)
print("EDA")
print("-"*80)
print(f"Vocabulary : \n{vocab}\n\nVocab size : {vocab_size}\n")

data = open("Dataset/names.txt").read().splitlines()
print(f"First ten samples before shuffling : \n{data[:10]}\n")
random.seed(13377)
random.shuffle(data)
print(f"First ten samples after shuffling : \n{data[:10]}\n")
max_ix = 0
for ix, name in enumerate(data):
    if len(name) > len(data[max_ix]):
        max_ix = ix
print(f"Longest input : {data[max_ix]}\t\tLength : {len(data[max_ix])}\n")
print("-"*80)


# Data preprocessing
stoi = {ch:i for i, ch in enumerate(vocab)}
itos = {i:ch for i, ch in enumerate(vocab)}
def encode(data):
    for i,name in enumerate(data):
        ix = []
        for ch in name:
            ix.append(stoi[ch])
        data[i] = torch.tensor(ix, dtype = torch.long)
        
decode = lambda ix: ''.join([itos[i] for i in ix])

def pad_sequences(data, pad_token, max_length,):
    for i,name in enumerate(data):
        if len(name) != max_length:
            pad_tensor = torch.full((max_length - len(name),), pad_token)
            data[i] = torch.cat((name, pad_tensor))

def split(data):
    n = int(0.9*len(data))
    xd = [d[:block_size] for d in data]
    yd = [d[1:] for d in data]
    xtr = torch.stack(xd[:n])
    ytr = torch.stack(yd[:n])
    xval = torch.stack(xd[n:])
    yval = torch.stack(yd[n:])
    return xtr, ytr, xval, yval

def get_batch(mode):
    if mode == "train":
        x = xtr
        y = ytr
    else:
        x = xval
        y = yval
    ix = torch.randint(len(x), (batch_size,))
    xb = x[ix]
    yb = y[ix]
    return xb, yb

encode(data)
pad_sequences(data, pad_token, max_length = 24)
xtr, ytr, xval, yval = split(data)
print(f"Train data size : {len(xtr)}\n")
print(f"Val data size : {len(xval)}\n")
print("-"*80)

@torch.no_grad()
def estimate_loss():
    m.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            attention_mask = xb==pad_token
            logits, loss = m(xb, attention_mask, yb)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    m.train()
    return out


# Model building
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x, attention_mask):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        att_sc = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        att_sc = att_sc.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        if attention_mask is not None:
            att_sc = att_sc.masked_fill(attention_mask.unsqueeze(1), float("-inf"))
        att_sc = F.softmax(att_sc, dim = -1)
        att_sc = self.dropout(att_sc)
        out = att_sc @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])   
        self.proj = nn.Linear(n_embd, n_embd, bias = False)
        self.dropout = nn.Dropout(dropout_ratio)
        
    def forward(self, x, attention_mask):
        out = torch.cat([h(x, attention_mask) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout_ratio)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa_head = MultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)

    def forward(self, x, attention_mask):
        x = x + self.sa_head(self.ln_1(x), attention_mask)
        x = x + self.ffwd(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block() for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias = False)

    def forward(self, x, attention_mask = None, targets = None):
        B, T = x.shape
        tok_emb = self.token_embedding_table(x)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index = pad_token)
        return logits, loss

    def generate(self, ix):
        while True:
            x = ix[:, -block_size:]
            logits, loss = self(x)
            logits = logits[:, -1, :]
            p_dis = F.softmax(logits, dim = -1)
            next_ix = torch.multinomial(p_dis, num_samples = 1)
            if next_ix == 3:
                return ix
            ix = torch.cat((ix, next_ix), dim = -1)

model = Transformer()
m = model.to(device)
print(f"Total parameters : {sum([p.nelement()for p in m.parameters()])} parameters\t{sum([p.nelement()for p in m.parameters()]) / 1e6:.2f}M parameters\n")
print("-"*80)


# Model training
optimizer = torch.optim.AdamW(m.parameters(), lr)
for iter in range(max_iters):
    if iter % eval_interval == 0:
        loss = estimate_loss()
        print(f"Step{iter} :\ttrain_loss : {loss['train']}\tval_loss : {loss['val']}")
    xb, yb = get_batch("train")
    xb, yb = xb.to(device), yb.to(device)
    attention_mask = xb==pad_token
    attention_mask = attention_mask.to(device)
    logits, loss = m(xb, attention_mask, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
print("-"*80)


#Inference
for i in range(12):
    ix = random.choice([2,56])
    input = torch.full((1,1), ix, device = device)
    out = m.generate(input)
    print(decode(out.tolist()[0]))
    
    
# Save the model
# Save the model state dictionary and necessary metadata
model_save_path = "NameGPT.pth"
torch.save({
    'model_state_dict': m.state_dict(),
    'hyperparameters': {
        'batch_size': batch_size,
        'block_size': block_size,
        'n_embd': n_embd,
        'n_heads': n_heads,
        'n_blocks': n_blocks,
        'dropout_ratio': dropout_ratio,
        'vocab_size': vocab_size,
        'pad_token': pad_token
    },
    'stoi': stoi,  # Save the string-to-index mapping
    'itos': itos,  # Save the index-to-string mapping
}, model_save_path)

print(f"Model saved to {model_save_path}")