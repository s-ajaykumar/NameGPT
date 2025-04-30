import random
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

torch.manual_seed(875745764)


# EDA
names =  open("Dataset/girl_names.txt").read().splitlines()
names[:10]
s = ""
for name in names:
   s += name
s = sorted(set(s))
print(f"Characters in each sample:\n{s}\n")
print(f"Total characters : {len(s)}\n")
print(f"Dataset length: {len(names)}")


# Data preprocessing
names_processed = []
chars = []
for name in names:
    name = name.lower()
    name = name.replace(' ', '')
    names_processed.append(name)

for name in names_processed:
    for ch in name:
        chars.append(ch)
chars = sorted(set(chars))
print(f"Characters in each sample:\n{chars}\n")
print(f"Total characters : {len(chars)}\n")
names_processed[:10]
random.seed(11739827)
random.shuffle(names_processed)
names_processed[:10]
stoi = {}
itos = {}
for i, char in enumerate(chars):
    stoi[char] = i+1
    itos[i+1] = char
stoi['.'] = 0
itos[0] = '.'
def process_dataset(dataset, block_size, stoi):
    xtr = []
    ytr = []
    for name in dataset:
        context = [0] * block_size
        name += '.'
        for ch in name:
            ix = stoi[ch]
            xtr.append(context)
            ytr.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(xtr), torch.tensor(ytr)
n1 = int(0.8 * len(names_processed))
n2 = int(0.9 * len(names_processed))
print(n1, n2)
xtr, ytr = process_dataset(names_processed[:n1], 3, stoi)
xval, yval = process_dataset(names_processed[n1:n2], 3, stoi)
xtest, ytest = process_dataset(names_processed[n2:], 3, stoi)
print(f"x-Training set length: {len(xtr)}")
print(f"y-Training set length: {len(ytr)}")
print(f"x-validation set length: {len(xval)}")
print(f"y-validation set length: {len(yval)}")
print(f"x-test set length: {len(xtest)}")
print(f"y-test set length: {len(ytest)}")


# Model building
class Linear:
    def __init__(self, fan_in, fan_out, bias = True):
        self.weight = torch.randn(fan_in, fan_out) / (fan_in) ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None
        
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias 
        return self.out

    def parameters(self):
        return [self.weight, self.bias] if self.bias is not None else [self.weight]


class BatchNorm1d:
    def __init__(self, dim, momentum = 0.001, eps = 4e-5, Training = True):
        self.eps = eps
        self.momentum = momentum
        self.Training = Training
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.mean_running = torch.zeros(dim)
        self.var_running = torch.zeros(dim)

    def __call__(self, x):
        if self.Training:
            xmean = x.mean(0, keepdims = True)
            xvar = x.var(0, keepdims = True)
        else:
            xmean = self.mean_running
            xvar = self.var_running
            
        self.out = self.gamma * (x - xmean) / torch.sqrt(xvar + self.eps) + self.beta
        
        if self.Training:
            with torch.no_grad():
                self.mean_running = (1 - self.momentum) * self.mean_running + (self.momentum * xmean)
                self.var_running = (1 - self.momentum) * self.var_running + (self.momentum * xvar)
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    def __init__(self, vocab_size, n_embd):
        self.C = torch.randn(vocab_size, n_embd)

    def __call__(self, x):
        self.out = self.C[x]   
        self.out = self.out.view(self.out.shape[0], -1)
        return self.out

    def parameters(self):
        return [self.C]


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    
block_size = 3
n_embd = 10
n_hidden = 100
vocab_size = 27
model = Sequential([
    Embedding(vocab_size, n_embd),
    
    Linear(block_size*n_embd, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),
    
    Linear(n_hidden, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),

    Linear(n_hidden, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),

    Linear(n_hidden, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),

    Linear(n_hidden, n_hidden),
    BatchNorm1d(n_hidden),
    Tanh(),
    
    Linear(n_hidden, vocab_size),
    BatchNorm1d(vocab_size)
])

with torch.no_grad():
    model.layers[-2].weight *= 0.1
    for layer in model.layers:
        if isinstance(layer, Linear):
            layer.weight *= 4/3
            
parameters = model.parameters()
for p in parameters:
    p.requires_grad = True
    
print(f"Total parameters : {sum([p.nelement() for p in parameters])}")


# Model training
batch_size = 32
loss_ls = []
for layer in model.layers:
    layer.Training = True
    
for i in range(200000):
    #Create mini batches
    ix = torch.randint(0, xtr.shape[0], (batch_size, ))
    x = xtr[ix]

    #Forward pass
    logits = model(x)
    loss = F.cross_entropy(logits, ytr[ix])
    
    #Backward pass
    for layer in model.layers:
        layer.out.retain_grad()
    for p in parameters:
        p.grad = None
    loss.backward()

    #Update
    if i < 100000:
        lr = 0.1
    else:
        lr = 0.01
    for p in parameters:
        p.data += -lr * p.grad

    #Track stats
    loss_ls.append(loss.item())
loss_ls = torch.tensor(loss_ls)
ls = []
loss_ls = loss_ls.view(-1, 1000).mean(1, keepdims = True)
for loss in loss_ls:
    ls.append(loss.item())
plt.plot(ls)
ls[-5:]


# EMA (Exploratory Model Analysis)
#Check activation distribution
plt.imshow(abs(model.layers[3].out) > 0.99, cmap = "gray", interpolation = "nearest")
plt.figure(figsize = (12,3))
for layer in model.layers:
    if isinstance(layer, Tanh):
        counts, g = torch.histogram(layer.out.grad, density = True)
        plt.plot(g[:-1].detach(), counts.detach())
plt.title("Activation - Gradient distribution")
plt.xlabel("Counts")
plt.ylabel("Activation gradients")
plt.figure(figsize = (18,4))
legends = []
for p in parameters:
    if p.ndim == 2:
        print(f"Update - Data Ratio: {lr*p.grad.std() / p.data.std().log10()}")
        legends.append(tuple(p.shape))
        counts, g = torch.histogram(p)
        plt.plot(g[:-1].detach(), counts.detach())

plt.title("Parameter - Gradient distribution")
plt.xlabel("Parameter gradients")
plt.ylabel("Counts")
plt.legend(legends)

for p in parameters:
    print(p.ndim)
for p in parameters:
    if p.ndim > 1:
        print(p.shape)
        print((0.1 * p.abs().std() / p.grad.abs().std().log()))
        
        
# Model evaluation
for layer in model.layers:
    layer.Training = False
x = xval
logits = model(x)
print(f"Validation loss : {F.cross_entropy(logits, yval).item()}")
# 3127 parameter model - train_loss : 2.50 - val_loss : 2.49
# 13000 parameter model + nembd : 1 - train_loss : 2.43 - val_loss: 2.44
# 14000 parmater model + nembd : 3 - train_loss : 1.76 - val_loss : 1.77
# 46497 parmater model + nembd : 10 - train_loss : 1.63 - val_loss : 1.73


# Inference
torch.manual_seed(875745764)
for layer in layers:
    layer.Training = False
for i in range(10):
    context = [0,0,0]
    while True:
        with torch.no_grad():
            x = torch.tensor([context])
            for layer in layers:
                x = layer(x)
            p_dis = F.softmax(x, dim = -1)
            ix = torch.multinomial(p_dis, num_samples = 1)
            if ix == 0:
                break
            print(itos[ix.item()], end = '')
            context = context[1:] +  [ix.item()]
    print()
    
    
'''model2:
-------------------------------
waabphiarhe

sre
ka
fkiasmahuia
noiyhieamarirhe
shnnanasuha
jaisha
fj
iaeenhaiiueazerbshinei

model 3:
-------------------------------
akanatina
vithra
karaksasmiga
dhaniyateamaritheeshna
sasha
deeksha
famithya
arunesharbalinee
nihankulee
vopmiya

model 4:
-------------------------------
akanitharmesha
risha
keasmithulanika
teasani
sembhana
ashri
jana
safa
ileenaayinesharshini
pranandhuleka'''