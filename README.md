# NameGPT

**NameGPT** is a GPT and GPT2 - based model designed to generate male and female names.<br>
Implemented it using pytorch.

## Goal
To create unique tamil names.

## Data collection
Scraped the internet using Selenium 

## Data preprocessing 
Built a tokenizer using BPE(Byte Pair Encoding) algorithm and the data collected.<br>
Tokenized the data.<br>
Added two different start tokens '~', '!' - one for boy names, another for girl names.<br>
Added an end token '.' and padded the inputs with a pad_token 

## Model building 
Implemented both the GPT and GPT2 architecture using Pytorch from scratch.<br>
Trained with following hyperparameters:

### Hyperparameters
batch_size      = 32<br>
block_size      = 23<br>
n_embd          = 384<br>
n_heads         = 6<br>
n_blocks        = 6<br>
dropout_ratio   = 0.2<br>
lr              = 3e-4<br>
max_iters       = 5001<br>
eval_interval   = 500<br>
eval_iters      = 200<br>
pad_token       = 57

## Model validation
val_loss        = 1.51<br>

## Tech stack
Implemented Gradio for UI