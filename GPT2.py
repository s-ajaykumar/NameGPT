# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class Config:
    batch_size : int
    n_embd : int
    n_heads : int
    n_layers : int
    dropout_ratio : float
    pad_token : int
    device : str
config = Config()
    
    
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
        att_sc = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.shape[-1]))
        att_sc = att_sc.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att_sc = F.softmax(att_sc, dim = -1)
        out = att_sc @ v
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
            
            
# Function to load model and generate a name
def load_model(model_path):
    global vocab_size, block_size, n_embd, n_heads, n_layers, dropout_ratio, pad_token, device, m, decode
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract hyperparameters
    hyperparameters = checkpoint['hyperparameters']
    config.vocab_size = hyperparameters['vocab_size']
    config.block_size = hyperparameters['block_size']
    config.n_embd = hyperparameters['n_embd']
    config.n_heads = hyperparameters['n_heads']
    config.n_layers = hyperparameters['n_layers']
    config.dropout_ratio = hyperparameters['dropout_ratio']
    config.pad_token = hyperparameters['pad_token']

    # Load stoi and itos
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    decode = lambda ix: ''.join([itos[i] for i in ix])

    # Initialize and load model
    model = GPT()
    model.load_state_dict(checkpoint['model_state_dict'])
    m = model.to(device)
    m.eval()

def generate_name(gender):
    # Generate a name
    start_ix = 56 if gender == "male" else 1  #
    input = torch.full((1, 1), start_ix, device=device, dtype=torch.long)
    out = m.generate(input)
    return decode(out.tolist()[0][1:])