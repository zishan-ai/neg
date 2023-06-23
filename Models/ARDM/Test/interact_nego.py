import os
import time
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

def top_p_candidates(logits, prob=0.92, filter_value=-float('Inf')):
    
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cum_sum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cum_sum > prob
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter_(1, index=sorted_indices, src=sorted_indices_to_remove.clone())
    logits[indices_to_remove] = filter_value
    
    return logits

# load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model_A = GPT2LMHeadModel.from_pretrained("gpt2")
model_B = GPT2LMHeadModel.from_pretrained("gpt2")

device = torch.device("cuda")
model_A = model_A.to(device)
model_B = model_B.to(device)
model_A_states = torch.load("model_v1_full_A5700.pth")
model_B_states = torch.load("/model_v1_full_B5700.pth")
model_A.load_state_dict(model_A_states)
model_B.load_state_dict(model_B_states)

model_A.eval()
model_B.eval()
sep = [628, 198]

temperature = 0.8
top_k = 400
top_p = 0.9
past = None
probs = 0.9
user = input("A:")
user = tokenizer.encode("A:"+user)
prev_input = user + sep
prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(device)
while(True):
    sent = []
    with torch.no_grad():
        for i in range(200):
            logits, past = model_A(prev_input, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_p_candidates(logits,top_p=0.9)
            probs = F.softmax(logits, -1)
            prev_input = torch.multinomial(probs, num_samples=1)
            prev_word = prev_input.item()

            if prev_word == 628:
                break
            else:
                sent.append(prev_word)
            
            # past_position_ids = past_position_ids[:, -1:] + 1


    print(tokenizer.decode(sent))
    _,past = model_B(prev_input,past)
    user = input("A:")
    if user == "quit":
        break
    user = tokenizer.encode("A:"+user)
    prev_input = user + sep
    prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(device)


