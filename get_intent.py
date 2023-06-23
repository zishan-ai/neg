import torch
import numpy as np
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
import pandas as pd
from torch.optim import Adam
from tqdm import tqdm
from torch import nn


class BertClassifier_buyer(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier_buyer, self).__init__()
        output_model_file = "/3.pth"
        model_state_dict = torch.load(output_model_file)
        self.bert = BertModel.from_pretrained('bert-base-uncased',state_dict = model_state_dict)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 12)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer
		
		
	    
		
    

    def get_prediction(self,curr_sentence,idx):
        label1 = 6
        label1 = torch.tensor(label1).to('cuda')
        model = BertClassifier_buyer().to('cuda')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #print('curr_sentence:')
        #print(curr_sentence)
        bert_input = tokenizer(curr_sentence,padding='max_length', max_length = 200, 
                       truncation=True, return_tensors="pt")
        #print(bert_input)
        bert_input_id = bert_input['input_ids'].squeeze(1).to('cuda')
        #print(bert_input_id)
        model.eval()
        bert_input_mask = bert_input['attention_mask'].to('cuda')
        with torch.no_grad():
            output = model(bert_input_id,bert_input_mask)
          
            
            predicted_probabilities = torch.softmax(output, dim=1).squeeze(0).tolist()
            #print('pred_prob',predicted_probabilities)
            #prediction = output.pooler_output.argmax(dim=1).item()
            #print(prediction)
            #print()
            return predicted_probabilities[idx]
            #label = self.tokenizer.decode(prediction)
        #pred_label_val = logits.item()
        #return pred_label_val[idx]
	
	
class BertClassifier_seller(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier_seller, self).__init__()
        output_model_file = "/3.pth"
        model_state_dict = torch.load(output_model_file)
        self.bert = BertModel.from_pretrained('bert-base-uncased',state_dict = model_state_dict)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 13)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer
		
		
	    
		
    

    def get_prediction(self,curr_sentence,idx):
        label1 = 6
        label1 = torch.tensor(label1).to('cuda')
        model = BertClassifier_seller().to('cuda')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #print('curr_sentence:')
        #print(curr_sentence)
        bert_input = tokenizer(curr_sentence,padding='max_length', max_length = 200, 
                       truncation=True, return_tensors="pt")
        #print(bert_input)
        bert_input_id = bert_input['input_ids'].squeeze(1).to('cuda')
        #print(bert_input_id)
        model.eval()
        bert_input_mask = bert_input['attention_mask'].to('cuda')
        with torch.no_grad():
            output = model(bert_input_id,bert_input_mask)
          
            
            predicted_probabilities = torch.softmax(output, dim=1).squeeze(0).tolist()
            #print('pred_prob',predicted_probabilities)
            #prediction = output.pooler_output.argmax(dim=1).item()
            #print(prediction)
            #print()
            return predicted_probabilities[idx]
            #label = self.tokenizer.decode(prediction)
        #pred_label_val = logits.item()
        #return pred_label_val[idx]
	
	