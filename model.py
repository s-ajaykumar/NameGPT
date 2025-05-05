# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the model classes (Head, MultiHeadAttention, FeedForward, Block, Transformer)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
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
        att_sc = F.softmax(att_sc, dim=-1)
        att_sc = self.dropout(att_sc)
        out = att_sc @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x, attention_mask):
        out = torch.cat([h(x, attention_mask) for h in self.heads], dim=-1)
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
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x, attention_mask=None, targets=None):
        B, T = x.shape
        tok_emb = self.token_embedding_table(x)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
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
            loss = F.cross_entropy(logits, targets, ignore_index=pad_token)
        return logits, loss

    def generate(self, ix):
        while True:
            x = ix[:, -block_size:]
            logits, _ = self(x)
            logits = logits[:, -1, :]
            p_dis = F.softmax(logits, dim=-1)
            next_ix = torch.multinomial(p_dis, num_samples=1)
            if next_ix == 3:  # Assuming 3 is the end token
                return ix
            ix = torch.cat((ix, next_ix), dim=-1)

# Function to load model and generate a name
def load_model_and_generate_name(model_path, gender):
    global vocab_size, block_size, n_embd, n_heads, n_blocks, dropout_ratio, pad_token, device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract hyperparameters
    hyperparameters = checkpoint['hyperparameters']
    vocab_size = hyperparameters['vocab_size']
    block_size = hyperparameters['block_size']
    n_embd = hyperparameters['n_embd']
    n_heads = hyperparameters['n_heads']
    n_blocks = hyperparameters['n_blocks']
    dropout_ratio = hyperparameters['dropout_ratio']
    pad_token = hyperparameters['pad_token']

    # Load stoi and itos
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    decode = lambda ix: ''.join([itos[i] for i in ix])

    # Initialize and load model
    model = Transformer()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Generate a name
    start_ix = 57 if gender == "male" else 2  #
    input = torch.full((1, 1), start_ix, device=device, dtype=torch.long)
    out = model.generate(input)
    return decode(out.tolist()[0][1:])