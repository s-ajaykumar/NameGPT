'''
My experiment is in lines. Residual connections are powerful because input is add to the output of each block. It's
like reminding the input at each block. The input does a huge change in a mlp beacuase it dot products with various
number of neurons. So what if we don't make a huge change to the input? So i modified the mlp block such that 
the out is calculate as usual but I only take the max value of output and multiply it with the input(x).
So that at each mlp layer only one transformation is done to the input. 
'''









import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


random.seed(1337)
torch.manual_seed(1337)


@dataclass
class Config:
    batch_size = 4
    n_embd = 126
    n_heads = 6
    n_layers = 6
    dropout_ratio = 0.2
    lr = 3e-4
    max_iters = 3001
    eval_interval = 300
    eval_iters = 200
    pad_token = 56
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = Config()
   
   
def encode(data):
    for i,name in enumerate(data):
        ix = []
        for ch in name:
            ix.append(stoi[ch])
        data[i] = torch.tensor(ix, dtype = torch.long)
    return data
        
        
decode = lambda ix: ''.join([itos[i] for i in ix])


def pad_sequences(data, pad_token, max_length):
    for i,name in enumerate(data):
        if len(name) != max_length:
            pad_tensor = torch.full((max_length - len(name),), pad_token)
            data[i] = torch.cat((name, pad_tensor))
    return data


def split(data):
    n = int(0.9*len(data))
    xd = [d[:config.block_size] for d in data]
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
    ix = torch.randint(len(x), (config.batch_size,))
    xb = x[ix]
    yb = y[ix]
    xb, yb = xb.to(config.device), yb.to(config.device)
    return xb, yb 


@torch.no_grad()
def estimate_loss():
    m.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for i in range(config.eval_iters):
            xb, yb = get_batch(split)
            logits, loss = m(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    m.train()
    return out


def train_model():
    optimizer = torch.optim.AdamW(m.parameters(), config.lr)
    for iter in range(config.max_iters):
        if iter % config.eval_interval == 0:
            loss = estimate_loss()
            print(f"Step{iter} :\ttrain_loss : {loss['train']}\tval_loss : {loss['val']}")
        xb, yb = get_batch("train")
        xb, yb = xb.to(config.device), yb.to(config.device)
        optimizer.zero_grad()
        logits, loss = m(xb, yb)
        loss.backward()
        optimizer.step()
    
    
# Model Building
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd, bias = False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = False)
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        self.c_proj.res_flag = 1
        
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(config.n_embd, dim = -1)
        q = q.view(B, T, config.n_heads, C//config.n_heads).transpose(1, 2)
        k = k.view(B, T, config.n_heads, C//config.n_heads).transpose(1, 2)
        v = v.view(B, T, config.n_heads, C//config.n_heads).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal = True)
        out = out.transpose(1, 2).contiguous().view(B, T, C) 
        out = self.c_proj(out)
        return out


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias = False)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias = False)
        self.c_proj.res_flag = 1

    def forward(self, x):
        out = self.c_fc(x)
        out = self.gelu(out)
        out = self.c_proj(out)
        #max_fired = torch.max(out, dim = -1).values 
        #out = x * max_fired.unsqueeze(2)
        return out


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = CausalSelfAttention()
        self.mlp = MLP()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block() for _ in range(config.n_layers)]),
            ln = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'res_flag'):
                std *= (2 * config.n_layers ** -0.5)
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
        
    def forward(self, x, targets = None):
        B, T = x.shape
        pos = torch.arange(0, T, dtype = torch.long, device = config.device)
        tok_emb = self.transformer.wte(x)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            logits = logits.view(-1, logits.shape[-1])
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets, ignore_index = config.pad_token)
        return logits, loss
    
    def generate(self, x):
        while True:
            logits, loss = m(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim = -1)
            ix = torch.multinomial(topk_probs, num_samples = 1)
            xcol = torch.gather(topk_indices, -1, ix)
            if xcol == 2:
                return x
            x = torch.cat((x, xcol), dim = -1)
        
    
# EDA
print("-"*80)
print("EDA")
print("-"*80)

text = open("Dataset/names.txt").read()

vocab = sorted(set(text))
config.vocab_size = len(vocab)
print(f"Vocabulary : \n{vocab}\n\nVocab size : {config.vocab_size}\n")

data = open("Dataset/names.txt").read().splitlines()

print(f"First ten samples before shuffling : \n{data[:10]}\n")
random.shuffle(data)
print(f"First ten samples after shuffling : \n{data[:10]}\n")

max_ix = 0
for ix, name in enumerate(data):
    if len(name) > len(data[max_ix]):
        max_ix = ix
longest_input_size = len(data[max_ix])
config.block_size = longest_input_size - 1
print(f"Longest input : {data[max_ix]}\t\tLength : {longest_input_size}\n")


# Data preprocessing
print("-"*80)
print("Data preprocessing started...")
print("-"*80)  
stoi = {ch:i for i, ch in enumerate(vocab[1:])}
itos = {i:ch for i, ch in enumerate(vocab[1:])}
data = encode(data)
data = pad_sequences(data, config.pad_token, max_length = longest_input_size)
xtr, ytr, xval, yval = split(data)
print(f"Train data size : {len(xtr)}\n")
print(f"Val data size : {len(xval)}\n")


# Model initialization
print("-"*80)
print("Model initializing...")
print("-"*80)    
model = GPT()
m = model.to(config.device)
#m = torch.compile(m)
print(f"Total parameters : {sum([p.nelement()for p in m.parameters()])} parameters\t{sum([p.nelement()for p in m.parameters()]) / 1e6:.2f}M parameters\n")


# Model training
print("-"*80)
print("Training started...")
print("-"*80)
train_model()


# Inference
random.seed(1337)
torch.manual_seed(1337)
print("-"*80)
print("Inference started...")
print("-"*80)
max_gen = 10
for _ in range(max_gen):
    ix = random.choice([55, 1])
    input = torch.full((1,1), ix, device = config.device)
    out = m.generate(input)
    print(decode(out.tolist()[0]))
print("-"*80)



