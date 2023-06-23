import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='GPU_NUMBER'

import numpy as np
import torch.nn as nn
from nltk.translate.meteor_score import meteor_score
from nltk.translate import meteor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import functools
import operator
import os
import pdb
import spacy
import pandas as pd
import json
import tqdm
import datetime
from tqdm.notebook import tqdm_notebook
from collections import Counter
from nltk import word_tokenize
from get_intent import BertClassifier_buyer,BertClassifier_seller
import random
import math
import pdb
from rlutils_nsr import collect_samples, ppo_step, generate_n_candidates, convert_sentences_to_strings, expand_inputs_for_N_candidates
from torch.utils.data import DataLoader, Dataset
from loss import SequenceCrossEntropyLoss
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from ppo import PPOMemory
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup#, RobertaForSequenceClassification, RobertaTokenizer
# from simpletransformers.classification import ClassificationModel,    
#torch.cuda.empty_cache()
#torch.autograd.set_detect_anomaly(True)
#from dataset import TopicShiftDataset
#from persona_nli import *
import warnings
warnings.filterwarnings("ignore")

model = SentenceTransformer('bert-base-nli-mean-tokens')

class Trainer():

    def __init__(self,
                 modelname,
                 csvfile,
                 n_epochs,
                 print_every,
                 learning_rate,
                 epsilon,
                 human_reward,
                 average_sent_loss,
                 device,
                 num_candidates,
                 max_candidate_length,
                 top_p,
                 warmup_steps,
                 pad_token_id,
                 evaluate_every,
                 use_intent,
                 use_prize_gap,
                 use_nego_strategy,
                 use_surface_similarity,
                 mini_batch,
                 temperature,
                 use_recent_past,
                 recompute_log_prob,
                 gamma1,
                 gamma2,
                 gamma3,
                 gamma4,
                 train_single_model=False,
                 single_model_to_train=None,
                 loadModel=False,
                 batch_size=None,
                 loadFilenameA=None,
                 loadFilenameB=None,
                 seedvalue=10):

        self.seedvalue = seedvalue
        self.train_single_model = train_single_model
        self.single_model_to_train = single_model_to_train
        self.nlp = spacy.load("en_core_web_sm")
        self.human_reward = human_reward
        self.seed(seedvalue)
        self.use_recent_past = use_recent_past
        self.temperature=temperature
        self.use_intent = use_intent
        self.use_prize_gap = use_prize_gap
        self.use_nego_strategy = use_nego_strategy
        self.use_surface_similarity = use_surface_similarity

        self.average_sent_loss = average_sent_loss
        self.mini_batch = mini_batch
        self.evaluate_every = evaluate_every
        self.csvfile = csvfile
        self.modelname = modelname
        self.n_epochs = n_epochs
        self.print_every = print_every
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
        self.recompute_log_prob = recompute_log_prob
        self.num_candidates = num_candidates
        self.pad_token_id = pad_token_id
        self.max_candidate_length = max_candidate_length
        
        
        self.top_p = top_p
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size

        self.device = device
        
        self.loadModel = loadModel
        self.loadFilenameA = loadFilenameA
        self.loadFilenameB = loadFilenameB
        self.make_model_save_dir()
        self.make_stats_dir()
        

        self.getDataset()
        
        self.initialize_models()
        self.configure_optimizer()
        
        self.buffer_memory = PPOMemory()
        
        self.saveModelConfig()
        self.criterion = SequenceCrossEntropyLoss()

        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.gamma4 = gamma4



    def saveModelConfig(self):
        if self.train_single_model:
            config_model_train = self.single_model_to_train
            print('Training Only :', self.single_model_to_train)
        else:
            config_model_train = 'Both Models being Trained.'
            print('Both Models being Trained.')
        config = {'Basic Info': [datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S")],
                  'NOTES': 'GPT2-MEDIUM',
                  'modelname': self.modelname,
                  'Training only one Model': self.train_single_model,
                  'Training Models': config_model_train,
                  'device': 'cuda',
                  'use_surface_similarity': self.use_surface_similarity,
                  'use_intent': self.use_intent,
                  'use_prize_gap': self.use_prize_gap,
                  'use_nego_strategy' : self.use_nego_strategy,
                  'modelLoaded': self.loadFilenameA,
                  'human_reward': self.human_reward,
                  'average_sent_loss' : self.average_sent_loss,
                  'n_epochs': self.n_epochs,
                  'use_recent_past': self.use_recent_past,
                  'temperature': self.temperature,
                  'learning_rate': self.learning_rate,
                  'epsilon': self.epsilon,
                  'num_candidates': self.num_candidates,
                  'pad_token_id': self.pad_token_id,
                  'max_candidate_length': self.max_candidate_length,
                  'recompute_log_prob': self.recompute_log_prob,
                  'evaluate_every': self.evaluate_every,
                  'top_p': self.top_p,
                  'warmup_steps': self.warmup_steps,
                  'batch_size':self.batch_size,
                  'seed': self.seedvalue}
        configfilename = os.path.join(self.savefolder, self.modelname, 'config')
        if not os.path.exists(configfilename):
            os.makedirs(configfilename)
        configfilename = configfilename + '/config' + '_' + self.modelname + '.json'
        with open(configfilename, 'w') as f:
            json.dump(config, f)
        #config.toJSON()
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)
    def seed(self,seed=10):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def extract_data(self, csvfile):
        df_dialogs = pd.read_csv(csvfile)
        data = {}
        next_utter={}
        next_intent = {}
        prev_buyer = 0.0
        prev_seller = 0.0
        j=0
        next_c_id = -99
        for i in tqdm.trange(len(df_dialogs)):
            line = df_dialogs.iloc[i]
            data_line = []
            
            #print(i)
            
            if line["conversation_id"] not in data:
                data[line["conversation_id"]] = []
                curr_c_id = line["conversation_id"]
                
                next_utter[line["conversation_id"]] = []
                next_intent[line["conversation_id"]] = []
                prev_buyer = 0.0
                prev_seller = 0.0

            if line["speaker"] == 0:
                
                #print("Buyer")
                text = 'A:'+"buyer to seller: '" +line["utterance"].strip() + "' with intent of "
                text = text + line['intent'].strip()
                if int(line['turn_no'])==1:
                    text = text + ", product description is "+ line['background_data'].strip()
                    text = text + ', buyer asking for: '+ line['items'].strip()
                else:
                    text = text + ', buyer asking for: '+ line['items'].strip()
                
                
                
                if line['price'] !='NA':
                    text = text + ',will offer #price $' + line['price'].strip() + ' , '
                    prev_buyer = line['price']
                else:
                    text = text + ' ,will offer #price $' + prev_buyer.strip()  + ' , '
                
               
                
            else:
                
                #print("seller")
                #print("speaker:",line["speaker"])
                text = "B:Seller to buyer: '" +line["utterance"].strip() + "' with intent of "
                text = text + line['intent'].strip() 
                
                text = text + ', seller telling about '+ line['items'].strip()
                
                
                if line['price'] !='NA':
                    text = text + ' ,will offer #price $' + line['price'].strip() + ' , '
                    prev_seller = line['price']
                else:
                    text = text + ' ,will offer #price $' + prev_seller.strip() + ' , '
                
                
            data[line["conversation_id"]].append(text)
            
            if i <= len(df_dialogs)-2:
                #print('yes')
                n_line =  df_dialogs.iloc[i+1]
                next_c_id = n_line["conversation_id"]
            if next_c_id!=curr_c_id:
                next_utter[line["conversation_id"]].append("<|end|>")
                next_intent[line["conversation_id"]].append("<|end|>")
            else:
                if i <= len(df_dialogs)-2:
                    #print(next_c_id)
                    #print(curr_c_id)
                    line_next = df_dialogs.iloc[i+1]
                    next_utter[line["conversation_id"]].append(line_next["utterance"])
                    next_intent[line["conversation_id"]].append(line_next['intent'])
                else:
                    next_utter[line["conversation_id"]].append("<|end|>")
                    next_intent[line["conversation_id"]].append("<|end|>")

            j+=1
                
        return data, next_intent, next_utter


        
    def utteranceToConversation(self, csvfile, data, persona, path, topic):
        df = pd.read_csv(self.csvfile)
        values=df['conv_id'].unique().tolist()
        conv_ids = df['conv_id'].tolist()

        dataset = []
        conversation = []
        personaset = []
        persona_conversation = []
        pathset = []
        path_conversation = []
        topicset = []
        topic_conversation = []
        for conv in values:
            for i in range(0, df.shape[0]):
                if(conv_ids[i]==conv):
                    conversation.append(data[i])
                    persona_conversation.append(persona[i])
                    path_conversation.append(path[i])
                    topic_conversation.append(topic[i])
                else:
                    continue
            dataset.append(conversation)
            personaset.append(persona_conversation)
            pathset.append(path_conversation)
            topicset.append(topic_conversation)
            conversation = []
            persona_conversation = []
            path_conversation = []
            topic_conversation = []
        return dataset, personaset, pathset, topicset

  
          
    def convertDicttoList(self, data: dict):
        return list(data.values())

    def random_split_data(self, data, next_intent, next_utter):
        indices = np.arange(len(data))
        np.random.shuffle(indices)

        train_data = [data[idx] for idx in indices[0:800]] # XXXX: nuber of dialogues in train dataset
        val_data = [data[idx] for idx in indices[800:1000]]
        train_seller_intent = [next_intent[idx] for idx in indices[0:800]]
        val_seller_intent = [next_intent[idx] for idx in indices[800:1000]]
        train_seller_utter = [next_utter[idx] for idx in indices[0:800]]
        val_seller_utter = [next_utter[idx] for idx in indices[800:1000]]
        
        return train_data, val_data, train_seller_intent, val_seller_intent, train_seller_utter, val_seller_utter


    def getDataset(self):
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        data, next_intent, next_utter = self.extract_data(self.csvfile)
        #data, persona, path, topic = self.utteranceToConversation(self.csvfile, data, persona, path, topic)
        ''' print(len(data),len(next_intent),len(next_utter))
        print("val_data")
        print(data[1])
        print(next_intent[1])
        print(next_utter[1])'''
        #self.traindata, self.valdata, self.traintopic, self.valtopic = train_data, val_data, train_topic, val_topic
        train_data, val_data, train_seller_intent, val_seller_intent, train_seller_utter, val_seller_utter = self.random_split_data(data, next_intent, next_utter)

        '''print("val_data")
        print(val_data[1])
        print(val_seller_intent[1])
        print(val_seller_utter[1])'''
        
        traindata_ = NegotiationDataset(train_data,
                                     train_seller_intent,
                                     train_seller_utter,
                                     self.tokenizer)

        
        self.turn_ending = traindata_.turn_ending
        
        
        valdata_ = NegotiationDataset(val_data,
                                     val_seller_intent,
                                     val_seller_utter,
                                     self.tokenizer)
        

        self.train_dataloader = DataLoader(dataset=traindata_,
                                           shuffle=True,
                                           batch_size=self.batch_size,
                                           collate_fn=traindata_.collate)
        
        self.val_dataloader = DataLoader(dataset=valdata_,
                                         shuffle=True,
                                         batch_size=self.batch_size,
                                         collate_fn=valdata_.collate)

    def initialize_models(self):
        if not self.train_single_model:
            self.model_A = GPT2LMHeadModel.from_pretrained("gpt2")
            self.model_A.to(device)
            
            self.model_B = GPT2LMHeadModel.from_pretrained("gpt2")
            self.model_B.to(device)
            self.model_A_ref = GPT2LMHeadModel.from_pretrained("gpt2")
            self.model_A_ref.to(device)
            self.model_B_ref = GPT2LMHeadModel.from_pretrained("gpt2")
            self.model_B_ref.to(device)
        else:
            if self.single_model_to_train == 'agent':
                self.model_A = GPT2LMHeadModel.from_pretrained("gpt2")
                self.model_A.to(device)
                self.model_A_ref = GPT2LMHeadModel.from_pretrained("gpt2")
                self.model_A_ref.to(device)
            else:
                self._model_B = GPT2LMHeadModel.from_pretrained("gpt2")
                self.model_B.to(device)
                self.model_B_ref = GPT2LMHeadModel.from_pretrained("gpt2")
                self.model_B_ref.to(device)

        if self.loadModel:
            if self.loadFilenameA:
                model_A_state_dict = torch.load(self.loadFilenameA, map_location=self.device)
                model_B_state_dict = torch.load(self.loadFilenameB, map_location=self.device)
                if not self.train_single_model:
                    self.model_A.load_state_dict(model_A_state_dict)
                    self.model_A_ref.load_state_dict(model_A_state_dict)
                    self.model_B.load_state_dict(model_B_state_dict)
                    self.model_B_ref.load_state_dict(model_B_state_dict)
                    self.model_A = self.model_A.to(self.device)
                    self.model_A_ref = self.model_A_ref.to(self.device)
                    self.model_B = self.model_B.to(self.device)
                    self.model_B_ref = self.model_B_ref.to(self.device)
                    self.model_A.train()
                    self.model_B.train()
                    self.model_A_ref.eval()
                    self.model_B_ref.eval()
                else:
                    if self.single_model_to_train == 'agent':
                        self.model_A.load_state_dict(model_A_state_dict)
                        self.model_A_ref.load_state_dict(model_A_state_dict)
                        self.model_A = self.model_A.to(self.device)
                        self.model_A_ref = self.model_A_ref.to(self.device)
                        self.model_A.train()
                        self.model_A_ref.eval()
                        #self.model_B.load_state_dict(model_B_state_dict) 
                        #self.model_B = self.model_B.to('cuda')
                        #self.model_B.eval()
                        self.model_B = None
                        self.model_B_ref = None
                    else:
                        self.model_B.load_state_dict(model_B_state_dict)
                        self.model_B_ref.load_state_dict(model_B_state_dict)
                        self.model_B = self.model_B.to(self.device)
                        self.model_B_ref = self.model_B_ref.to(self.device)
                        self.model_B.train()
                        self.model_B_ref.eval()
                        self.model_A = None
                        self.model_A_ref = None
                print('\n')
                print("Models loaded from file ", self.loadFilenameA)
            else:
                print('Models not loaded since directory not provided.')
        print(f"Models Initalized!")
        print('\n')


    def configure_optimizer(self):
        
        self.num_train_optimization_steps = self.n_epochs * 3000 # // self.batch_size  ### Hardcoded

        if not self.train_single_model:
            param_optimizer = list(self.model_A.named_parameters()) + list(self.model_B.named_parameters())
        else:
            if self.single_model_to_train == 'agent':
                param_optimizer = list(self.model_A.named_parameters())
        no_decay = ['bias', 'ln', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = optimizer = AdamW(optimizer_grouped_parameters,
                                           lr=self.learning_rate,
                                           eps=1e-06)

        #self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
        #                                                 num_warmup_steps=self.warmup_steps,
        #                                                 num_training_steps=self.num_train_optimization_steps)

        '''self.scheduler = WarmupLinearSchedule(self.optimizer,
                                                 warmup_steps=self.warmup_steps,
                                                 t_total=self.num_train_optimization_steps)'''


    def get_candidate_lengths(self, candidates):



        avg_iter_length = []
        
        for i in candidates:
            candidate_sentence = self.tokenizer.decode(i.tolist()[0][2:]).split('\n')[0].split('\t')[0]
            avg_iter_length.append(len(candidate_sentence.split()))
        return avg_iter_length

    '''def get_up_con_score(self, candidates, turn_num, dial_inputs):

        up_con_score = []
        for i in candidates:
            candidate = self.tokenizer.decode(i.tolist()[0][2:]).split('\n')[0].split('\t')[0]
            if(turn_num>=2):
                persona = self.tokenizer.decode(dial_inputs[turn_num-1].tolist()[0]).split('\n')[0].split('\t')[1].strip()
            else:
                persona = ''

            turn = []
            turn.append(persona)
            s = get_nli(candidate, turn)
            z = torch.argmax(s)
            if z==0:
                up_con_score.append(-1)
            elif z==2:
                up_con_score.append(1)
            else:
                up_con_score.append(0)

        return up_con_score'''

    def label2idx(self,role_id,pred_intent):
        
        buyer_intent = {
                'greet-ask':0,
                'negotiate-price-decrease':1,
                'negotiate-remove-x':2,
                'ask_clarification-y':3,
                'negotiate-price-nochange':4,
                'negotiate-add-x':5,
                'ask_price':6,
                'accept':7,
                'reject':8,
                'greet-ask_negotiate-price-decrease':9,
                'negotiate-remove-x_negotiate-price-decrease':10,
                'negotiate-remove-delivery':11,
            }
        seller_intent = {
                'greet-inform':0,
                'negotiate-price-nochange':1,
                'negotiate-price-increase':2,
                'tell_price':3,
                'provide_clarification-y':4,
                'negotiate-add-x':5,
                'accept':6,
                'greet-inform_negotiate-price-increase':7,
                'acknowledge acceptance':8,
                'greet-inform_negotiate-price-nochange':9,
                'negotiate-remove-delivery':10,
                'negotiate-price-remove-x':11,
                'avoid_rejection':12
            }
        if role_id==0:
            return buyer_intent[pred_intent]
        else:
            return seller_intent[pred_intent]
            

    def get_meteor_score(self, candidates, current_sentence):

        meteor_score_list = []
        
        for i in candidates:
            reference = []
            candidate = self.tokenizer.decode(i.tolist()[0][2:]).split('\n')[0].split('\t')[0]
            predicted = word_tokenize(candidate) 
            ref = word_tokenize(current_sentence)
            reference.append(ref)
            meteor_score = round(meteor(reference, predicted),2)  
            meteor_score_list.append(meteor_score)         
        return meteor_score_list 


    def get_utt_t_score(self, candidates, turn_num, dial_inputs):

        utt_t_list = []
        
        for i in candidates:
            candidate = self.tokenizer.decode(i.tolist()[0][2:]).split('\n')[0].split('\t')[0]
            if(turn_num>=2):
                topic = self.tokenizer.decode(dial_inputs[turn_num-1].tolist()[0]).split('\n')[0].split('\t')[3].strip()
            else:
                topic = ''
            turn = []
            turn.append(candidate)
            turn.append(topic)
            turn=model.encode(turn)
            score = cosine_similarity([turn[0]], turn[1:])[0][0]
            utt_t_list.append(score)
        
        return utt_t_list
    
    def intent_pred(self,current_sentence,pred_intent,role_id):
  
            intent_reward = []
            idx = -1
            print("Inside intent_pred_module")
            print(current_sentence[0])
            
            for i in range(len(current_sentence)):
                indx = current_sentence[i].find('next')
                current_sentence[i] = current_sentence[i][indx:]
            for i in range(len(current_sentence)):
                if pred_intent[i]!='NA':
                    idx = self.label2idx(pred_intent[i].lower(),role_id)
                if idx!=-1:
                    if role_id ==0:
                        #intent_re = 0.54
                        prediction = BertClassifier_buyer()
                        intent_re = prediction.get_prediction(current_sentence[i],idx)
                        intent_reward.append(intent_re)
                        #return intent_reward
                    if role_id == 1:
                        #intent_re=0.6
                        prediction = BertClassifier_seller()
                        intent_re = prediction.get_prediction(current_sentence[i],idx)
                        intent_reward.append(intent_re)
                else:
                    print('Not taking Intent')
                    intent_reward.append(0)
            return intent_reward

    def prize_gap(self,seller_ip,seller_final):
        res = []
        for i in range(len(seller_ip)):
            res.append(seller_final[i]/seller_ip[i])
        return res

    def neg_eval(self,prize_diff):
        if prize_diff>=0:
            #return 1/(1 + np.exp(-prize_diff))
            return np.exp(prize_diff)
        else:
            return 0
  
    def negotiation_result(self,result,buyer_price,seller_min_price):
        rfactor = 0
        val_ret = []
        for i in range(len(buyer_price)):
            if result == 'Accept':
                rfactor = 1
            else:
                rfactor = -1
            val_ret.append(self.neg_eval(((buyer_price[i] - seller_min_price[i])/(buyer_price[i]+1))*rfactor))
        return val_ret


    def sentence_surface_similarity(curr_sentence,context):
  
        cos_values = []
        sim_val = []
        print(curr_sentence)
        for sent in curr_sentence:
            if len(context)==0:
                sim_val.append(0)
            for past_sentence in context:
                trim_len = len(sent)
                con_len = len(past_sentence)
                if trim_len>con_len:
                    sent = sent[:con_len]
                A = sent
                B = past_sentence
            #B = "I would like to offer you laptop and charger in very good price"
                A_dic = Counter()
                B_dic = Counter()
                A = A.lower()
                B = B.lower()

                A_ls = A.split(" ")
                B_ls = B.split(" ")
                for item in A_ls:
                    A_dic[item] = A_dic[item]+1
                for item in B_ls:
                    B_dic[item] = B_dic[item]+1
                sop = 0
                sqrta = 0
                sqrtb = 0
                for key in A_dic.keys():
                    if B_dic[key]>0:
                    #print('yes')
                        sop = sop + B_dic[key]*A_dic[key]
                    #print(B_dic[key]*A_dic[key])
                    sqrta = sqrta + math.pow(A_dic[key],2)
                for key in B_dic.keys():
                    sqrtb = sqrtb + math.pow(B_dic[key],2)
                #print(sop,sqrta,sqrtb)
                cos_val = sop/(math.sqrt(sqrta)*math.sqrt(sqrtb))
                cos_values.append(cos_val)
            sim_val.append(sum(cos_values)/len(cos_values))
        return sim_val
    
    def top_p_candidates(self,logits, prob=0.92, filter_value=-float('Inf')):
    
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        cum_sum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cum_sum > prob
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter_(1, index=sorted_indices, src=sorted_indices_to_remove.clone())
        logits[indices_to_remove] = filter_value
        
        return logits

    def get_logits_output(self,logits):
        next_token_logits = logits[:, -1, :].contiguous() / 0.8
        next_token_logits = self.top_p_candidates(next_token_logits, 0.92)
        next_token_log_probs = F.log_softmax(next_token_logits, -1)
        probs = F.softmax(next_token_logits, dim=-1)
                
        next_tokens = torch.multinomial(probs, num_samples=1)
        next_token_log_probs = next_token_log_probs.gather(-1, next_tokens)
        next_tokens = next_tokens.squeeze(1)
        return next_tokens.cpu()
    def get_predicted_intent(self,sentence,role_id):
    
        buyer_intent = {
                    'Greet-Ask':0,
                    'Negotiate-Price-Decrease':1,
                    'Negotiate-Remove-X':2,
                    'Ask_Clarification-Y':3,
                    'Negotiate-Price-NoChange':4,
                    'Negotiate-Add-X':5,
                    'Ask_Price':6,
                    'Accept':7,
                    'Reject':8,
                    'Greet-Ask_Negotiate-Price-Decrease':9,
                    'Negotiate-Remove-X_Negotiate-Price-Decrease':10,
                    'Negotiate-Remove-delivery':11,
                }
        seller_intent = {
                    'Greet-Inform':0,
                    'Negotiate-Price-NoChange':1,
                    'Negotiate-Price-Increase':2,
                    'Tell_Price':3,
                    'Provide_Clarification-Y':4,
                    'Negotiate-Add-X':5,
                    'Accept':6,
                    'Greet-Inform_Negotiate-Price-Increase':7,
                    'Acknowledge-acceptance':8,
                    'Greet-Inform_Negotiate-Price-NoChange':9,
                    'Negotiate-Remove-delivery':10,
                    'Negotiate-Price-Remove-X':11,
                    'avoid_rejection':12
                }
        indx = sentence.find('next')
        sentence = sentence[indx:]
        sent = sentence.split(' ')
        sent = [tok.strip() for tok in sent if len(tok)>0]
        get_intent = False
        intent = ''
        
        for i in range(len(sent)):
            if sent[i] == 'intent:':
                get_intent = True
            if get_intent == True:
                intent = sent[i]
                get_intent = False
            if i<len(sent)-1:
                if sent[i]=='intent' and sent[i+1] == 'of':
                    get_intent = True
                if get_intent==True and sent[i]!='of':
                    intent = sent[i]
                    get_intent = False
        if role_id == 0:
            if intent in seller_intent.keys():
                return intent
            else:
                return 'NA'
        elif role_id==1:
            if intent in buyer_intent.keys():
                return intent
            else:
                return 'NA'


    def slice_input(self,dial_turn_input):
        slice_len = 0
        #print('shape of turn 1: ',dial_turn_input.shape[1])
        for i in range(dial_turn_input.shape[1]):
            if dial_turn_input[0][i] == 197:
                #print("sliced")
                #print("slice_len:",slice_len)
                #print("tensor: ",dial_turn_input[0])
                #print("if")
                return slice_len
            else:
                #print("tensor: ",dial_turn_input[0][i])
                #print("else")
                slice_len +=1
        return slice_len

    def validate_model(self, dataloader):
        print("Validation Step")
        f = open('exp1_output_sample_nsr_1000.txt','a')
        with torch.no_grad():
            if not self.train_single_model:
                self.model_A.eval()
                self.model_B.eval()
            else:
                if self.single_model_to_train == 'agent':
                    self.model_A.eval()
                else:
                    self.model_B.eval()

            with torch.no_grad():
                
                progress_bar = tqdm_notebook
                pbar = progress_bar(dataloader)
               
                total_ppl = []
                total_loss = []
                total_r_len = []
                total_meteor = []
                total_strategy_reward = []
                total_prize_gap_reward = []
                context = []
                for batch in pbar:
                    role_ids, _ = batch[0]
                    r_id = []
                    conv_list=[]
                    ind=0
                    #if sum([len(item) for item in batch[0][1]]) > 1024:
                    role_ids, _ = batch[0]
                    r_id = []
                    conv_list = []
                    ind=0
                    br_flag = False
                    last_item = len(batch[0][1])-1
                    last_rid = len(role_ids)-1
                    print("Last item, rid index",last_item,last_rid)
                    if sum([len(item) for item in batch[0][1]]) > 1024:
                        
                        print("#################### not working!!!!!!!!#####################")
                        #print(batch[0][1])
                        trim_indx = 0
                        #batch_ = []
                        print(type(batch[0][1]))
                        
                        for item in batch[0][1]:
                            conv = item
                            print("Added: ",ind)
                            
                            
                            trim_indx = trim_indx + len(conv)
                            if trim_indx<1024 - len(batch[0][1][last_item]):
                                conv_list.append(conv)
                                ind=ind+1
                            else:
                                br_flag = True
                                break

                        for i in range(ind):
                            r_id.append(role_ids[i])
                        if br_flag==True:
                            print("Last_item_added")
                            conv_list.append(batch[0][1][last_item])
                            r_id.append(role_ids[last_rid])
                        
                        batch[0] = (r_id,conv_list)


                    role_ids, dialog_tokens = batch[0]


                    dial_inputs = [torch.LongTensor(item).unsqueeze(0) for item in dialog_tokens]
                    past = None
                    past_ = None
                    all_logits = []
                    target = []
                    conv_sentences = []
                    final_conv = len(dial_inputs)-1
                    #print('conversation length:',final_conv)
                    i=0
                    final_intent = ''
                    seller_initial_price = 10000
                    init_price=False
                    seller_minimum_price=10000
                    for turn_num, dial_turn_inputs in enumerate(dial_inputs):
                        
                        current_sentence = self.tokenizer.decode(dial_turn_inputs.tolist()[0][2:]).split('\t')[0]
                        

                        if i==1 or init_price==False:
                            #print("******Current sentence")
                            #print(current_sentence)
                            curr_sent_word = current_sentence.split(' ')
                            get_price = False
                            for word in curr_sent_word:
                                
                                if word == '#price':
                                    #print('mark1')
                                    get_price = True
                                    continue
                                if get_price == True:
                                    word_ = word.replace(".","")
                                    word_ = word_[1:-3]
                                    if word_.isnumeric():
                                        if word[0]=='$':
                                            word = word[1:-3]
                                        #ab = "'23'"
                                        #print("word:",word.strip())
                                        #print(":",ab)
                                        seller_initial_price = int(float(word))
                                        init_price=True
                                        break
                                    else:
                                        init_price=False
                                        break
                                    

                        if init_price==True:           
                            seller_minimum_price = 0.7*seller_initial_price

                            
                        buyer_price = 0
                        prize_diff = 0
                        buyer_pr_flag=False
                        if i == final_conv:
                            #print("******Current sentence")
                            #print(current_sentence)
                            #print("mark3: ",i)
                            if role_ids[turn_num] == 0:
                                final_intent = 'Reject'
                                curr_sent_word = current_sentence.split(' ')
                                get_price = False
                                for word in curr_sent_word:
                                    if word == '#price':
                                        get_price = True
                                        continue
                                    if get_price == True:
                                        if word!= "NA'":
                                            if word[0]=='$':
                                                word = word[1:-3]
                                            buyer_price = int(float(word))
                                            buyer_pr_flag
                                            break
                            elif role_ids[turn_num]==1:
                                final_intent = 'Accept'
                            prize_diff = buyer_price - seller_minimum_price
                        if not self.train_single_model:
                            if role_ids[turn_num] == 0:
                                index = self.slice_input(dial_turn_inputs)
                                
                                new_input = dial_turn_inputs[:,0:index]
                                input_act = convert_sentences_to_strings([dial_turn_inputs], self.tokenizer)[0]
                                new_input_str = convert_sentences_to_strings([new_input], self.tokenizer)[0]
                                f.write('Actual Input')
                                f.write('\n')
                                f.write(input_act)
                                f.write('\n')
                                f.write('new_input')
                                f.write(new_input_str)
                                
                                dial_turn_inputs = dial_turn_inputs.to(self.device)
                                outputs = self.model_A(dial_turn_inputs, past_key_values=past, return_dict=False)
                                
                                index = self.slice_input(dial_turn_inputs)
                                new_input = dial_turn_inputs[:,0:index]
                                generated_sequence, generated_log_probs  = generate_n_candidates(self.model_A,
                                                                                          new_input,
                                                                                          self.top_p,
                                                                                          eos_token_id=self.turn_ending[0],
                                                                                          pad_token_id=self.turn_ending[1],
                                                                                          num_candidates=self.num_candidates,
                                                                                          max_gen_length=200,
                                                                                          temperature=self.temperature,
                                                                                          past=past_,
                                                                                          device=self.device)
                                #sent_logit = self.get_logits_output(outputs[0])
                                pred_sentence = convert_sentences_to_strings(generated_sequence, self.tokenizer)
                                p_sent = pred_sentence[0]
                                f.write('\n')
                                f.write('predicted_sentence')
                                f.write(p_sent)
                                pred_intent = []
                                buyer_p = []
                                seller_min_pr = []
                                seller_init_price = []
                                for sent in pred_sentence:

                                    pred_intent.append(self.get_predicted_intent(sent,role_ids[turn_num]))
                                    buyer_p.append(buyer_price)
                                    seller_min_pr.append(seller_minimum_price)
                                    seller_init_price.append(seller_initial_price)
                                #pred_intent = self.get_predicted_intent(pred_sentence,role_ids[turn_num])
                                #pred_intent = 'Accept'
                                past = outputs[1]
                                all_logits.append(outputs[0])
                            else:
                                
                                dial_turn_inputs = dial_turn_inputs.to(self.device)
                                outputs = self.model_B(dial_turn_inputs, past_key_values=past, return_dict=False)
                                index = self.slice_input(dial_turn_inputs)
                                new_input = dial_turn_inputs[:,0:index]
                                input_act = convert_sentences_to_strings([dial_turn_inputs], self.tokenizer)[0]
                                new_input_str = convert_sentences_to_strings([new_input], self.tokenizer)[0]
                                f.write('Actual Input')
                                f.write('\n')
                                f.write(input_act)
                                f.write('\n')
                                f.write('new_input')
                                f.write(new_input_str)
                                generated_sequence, generated_log_probs  = generate_n_candidates(self.model_A,
                                                                                          new_input,
                                                                                          self.top_p,
                                                                                          eos_token_id=self.turn_ending[0],
                                                                                          pad_token_id=self.turn_ending[1],
                                                                                          num_candidates=self.num_candidates,
                                                                                          max_gen_length=200,
                                                                                          temperature=self.temperature,
                                                                                          past=past_,
                                                                                          device=self.device)
                                #sent_logit = self.get_logits_output(outputs[0])
                                pred_sentence = convert_sentences_to_strings(generated_sequence, self.tokenizer)
                                #print("Validation output: \n",pred_sentence)
                                #pred_sentence = convert_sentences_to_strings(generated_sequence, self.tokenizer)
                                p_sent = pred_sentence[0]
                                f.write('\n')
                                f.write('predicted_sentence')
                                f.write(p_sent)
                                pred_intent = []
                                buyer_p = []
                                seller_min_pr = []
                                seller_init_price = []
                                for sent in pred_sentence:

                                    pred_intent.append(self.get_predicted_intent(sent,role_ids[turn_num]))
                                    buyer_p.append(buyer_price)
                                    seller_min_pr.append(seller_minimum_price)
                                    seller_init_price.append(seller_initial_price)
                                #pred_intent = self.get_predicted_intent(pred_sentence,role_ids[turn_num])
                                #pred_intent = 'Accept'
                                past = outputs[1]
                                all_logits.append(outputs[0])
                        else:
                            if self.single_model_to_train == 'agent':
                                if role_ids[turn_num] == 0:
                                    dial_turn_inputs = dial_turn_inputs.to(self.device)
                                    index = self.slice_input(dial_turn_inputs)
                                    new_input = dial_turn_inputs[:,0:index]
                                    input_act = convert_sentences_to_strings([dial_turn_inputs], self.tokenizer)[0]
                                    outputs = self.model_A(dial_turn_inputs, past_key_values=past, return_dict=False)
                                    #sent_logit = self.get_logits_output(outputs[0])
                                    #pred_sentence = convert_sentences_to_strings([sent_logit], self.tokenizer)[0]
                                    #pred_intent = self.get_predicted_intent(pred_sentence)
                                    past = outputs[1]
                                    all_logits.append(outputs[0])
                                    target.append(dial_turn_inputs)
                                    generated_sequence, generated_log_probs  = generate_n_candidates(self.model_A,
                                                                                          new_input,
                                                                                          self.top_p,
                                                                                          eos_token_id=self.turn_ending[0],
                                                                                          pad_token_id=self.turn_ending[1],
                                                                                          num_candidates=self.num_candidates,
                                                                                          max_gen_length=200,
                                                                                          temperature=self.temperature,
                                                                                          past=past_,
                                                                                          device=self.device)
                                    output = self.model_A(expand_inputs_for_N_candidates(dial_turn_inputs,
                                                                                         self.num_candidates),
                                                                                         past_,
                                                                                         return_dict=False)
                                    past_ = output[1]
                                    current_sentence = self.tokenizer.decode(dial_turn_inputs.tolist()[0][2:]).split('\t')[0]
                        i = i+1

                    all_logits = torch.cat(all_logits, dim=1)
                    all_logits = all_logits[:, :-1].contiguous()

                    if not self.train_single_model:
                        target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()
                    else:
                        target = torch.cat(target, dim=1)[:, 1:].contiguous()
                    
                    target_mask = torch.ones_like(target).float()

                    loss = self.criterion(all_logits.to('cuda'), target.to('cuda'), target_mask.to('cuda'), label_smoothing=-1, reduce='sentence')
                    total_loss.extend(loss.tolist())

                    ppl = torch.exp(loss)
                    total_ppl.extend(ppl.tolist())
                    


                    average_lengths = self.get_candidate_lengths(generated_sequence)
                    total_r_len.append(np.mean(average_lengths))

                    meteor_scores = self.get_meteor_score(generated_sequence, current_sentence)
                    total_meteor.append(np.mean(meteor_scores))

                    strategy_score = self.negotiation_result(final_intent, buyer_p,seller_min_pr)
                    total_strategy_reward.append(np.mean(strategy_score))

                    gap_score = self.prize_gap(seller_init_price,buyer_p)
                    total_prize_gap_reward.append(np.mean(gap_score))

                print('\n')
                print(f"Validation Perplexity: {np.mean(total_ppl)}")

                # average_lengths = self.get_candidate_lengths(generated_sequence)
                print(f"Overall Average candidate length: {np.mean(total_r_len)}")
                print(f"Overall Meteor score: {np.mean(total_meteor)}")
                print(f"Overall negotiation strategy score: {np.mean(total_strategy_reward)}")
                print(f"Overall maximum utility score: {np.mean(total_prize_gap_reward)}")
                context.append(current_sentence)

        return np.mean(total_ppl), np.mean(total_loss), np.mean(average_lengths)
    

    def make_stats_dir(self):
        
        self.statsfolder = os.path.join(os.getcwd(), self.savefolder, self.modelname, 'stats')
        if not os.path.exists(self.statsfolder):
            os.makedirs(self.statsfolder)


    def make_model_save_dir(self):
        
        self.savefolder = os.path.join(os.getcwd(), 'Path_to_save_the_trained_model_exp1_1')
        if not os.path.exists(self.savefolder):
            print("Model save folder doesn't exist.")
            os.makedirs(self.savefolder)
            print(f"Created folder {self.savefolder} to save the models.")


    def save_models(self, num_iter):
        
        modeldir = os.path.join(self.savefolder, self.modelname)
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)
            print('Created Directory for saving models!')
        filenameA = modeldir + '/' + self.modelname + '_A' + str(num_iter) + ".pth"
        filenameB = modeldir + '/' + self.modelname + '_B' + str(num_iter) + ".pth"
        #torch.save([self.model_A.state_dict(), self.model_B.state_dict()], filename)
        torch.save(self.model_A.state_dict(), filenameA)
        torch.save(self.model_B.state_dict(), filenameB)


    def modified_train_one_iter(self, batch):
        dial_inputs, role_ids, scores_dict = collect_samples(batch,
                                                             model_A=self.model_A_ref,
                                                             model_B=self.model_B,
                                                             top_p=self.top_p,
                                                             eos_token_id=self.turn_ending[0],
                                                             pad_token_id=self.turn_ending[1],
                                                             average_sent_loss=self.average_sent_loss,
                                                             max_gen_length=self.max_candidate_length,
                                                             buffer_memory=self.buffer_memory,
                                                             use_intent=self.use_intent,
                                                             use_prize_gap=self.use_prize_gap,
                                                             use_nego_strategy=self.use_nego_strategy,
                                                             use_surface_similarity=self.use_surface_similarity,
                                                             device=self.device,
                                                             num_candidates=self.num_candidates,
                                                             human_reward=self.human_reward,
                                                             tokenizer=self.tokenizer,
                                                             criterion=self.criterion,
                                                             temperature=self.temperature,
                                                             use_recent_past=self.use_recent_past,
                                                             recompute_log_prob=self.recompute_log_prob,
                                                             nlp=self.nlp,
                                                             train_single_model=self.train_single_model,
                                                             model_to_train=self.single_model_to_train,
                                                             gamma1=self.gamma1,
                                                             gamma2=self.gamma2,
                                                             gamma3=self.gamma3,
                                                             gamma4=self.gamma4)

        log_dict = ppo_step(model_A=self.model_A,
                            model_B=self.model_B,
                            buffer_memory=self.buffer_memory,
                            train_single_model=self.train_single_model,
                            dial_inputs= dial_inputs,
                            model_to_train=self.single_model_to_train,
                            device=self.device,
                            ppo_epsilon=self.epsilon,
                            num_candidates=self.num_candidates,
                            use_recent_past=self.use_recent_past,
                            average_sent_loss=self.average_sent_loss,
                            criterion=self.criterion,
                            optimizer=self.optimizer,
                            role_ids=role_ids)

        self.buffer_memory.clear_memory()

        return log_dict, scores_dict 
 
    def train(self):

        update_count = 0
        progress_bar = tqdm_notebook

        val_ppl = []
        val_loss = []

        rewards = []
        kl = []
        clip_frac = []

        intent_scores = []
        strategy_scores = []
        prize_gap = []
        sentence_similarity_scores = []
        

        best_ppl = None
        
        length = None
        
        iters = None
        
        #strategies = None
        progress_bar = tqdm.tqdm_notebook
        pbar = progress_bar(self.train_dataloader)

        for i in range(self.n_epochs):
            if not self.train_single_model:
                self.model_A.train()
                self.model_B.train()
            else:
                if self.single_model_to_train == 'agent':
                    self.model_A.train()
            
            for batch in pbar:
                role_ids, _ = batch[0]
                r_id = []
                conv_list = []
                ind=0
                br_flag = False
                last_item = len(batch[0][1])-1
                last_rid = len(role_ids)-1
                #print("Last item, rid index",last_item,last_rid)
                if sum([len(item) for item in batch[0][1]]) > 1024-300:
                    
                    print("#################### not working!!!!!!!!#####################")
                    #print(batch[0][1])
                    trim_indx = 0
                    #batch_ = []
                    #print(type(batch[0][1]))
                    
                    for item in batch[0][1]:
                        conv = item
                        #print("Added: ",ind)
                        
                        
                        trim_indx = trim_indx + len(conv)
                        if trim_indx<1024 - len(batch[0][1][last_item])-300:
                            conv_list.append(conv)
                            ind=ind+1
                        else:
                            br_flag = True
                            break

                    for j in range(ind):
                        r_id.append(role_ids[j])
                    if br_flag==True:
                        print("Last_item_added")
                        conv_list.append(batch[0][1][last_item])
                        r_id.append(role_ids[last_rid])
                    
                    batch[0] = (r_id,conv_list)


                print(f"ITERATION: {update_count}")
                print("Epoch: ",i)

                batch = batch[0]
                #print(batch)
                log_dict, scores_dict  = self.modified_train_one_iter(batch)

                clip_frac.append(log_dict['clip_frac'])
                kl.append(log_dict['approx_kl'])
                rewards.append(log_dict['reward'])

                intent_scores.extend(scores_dict['intent'])
                strategy_scores.extend(scores_dict['negotiation_strategy'])
                prize_gap.extend(scores_dict['prize_gap'])
                sentence_similarity_scores.extend(scores_dict['sentence_similarity'])
                

                np.save(self.statsfolder + '/' + 'intent_scores.npy', np.array(intent_scores))
                np.save(self.statsfolder + '/' + 'strategy_scores.npy', np.array(strategy_scores))
                np.save(self.statsfolder + '/' + 'prize_gap_scores.npy', np.array(prize_gap))
                np.save(self.statsfolder + '/' + 'sentence_similarity_scores.npy', np.array(sentence_similarity_scores))
                
                update_count += 1
#y

                print('update count is:', update_count)

                if  update_count % self.evaluate_every == 0:
                    
                    ppl, loss, average_length = self.validate_model(self.val_dataloader)
                    
                    if best_ppl is None:

                        best_ppl = ppl
                        iters = update_count
                        
                        length = average_length
                        
                        
                        self.save_models(iters)
                        print(f'Saving Model at {iters}')
                        
                    else:
                        if ppl < best_ppl:
                            best_ppl = ppl
                            iters = update_count
                            
                            length = average_length
                            
                        
                            self.save_models(iters)
                            print(f'Saving Model at {iters}')
                
                    print('\n')
                    print(f'Best Perplexity Found so far {best_ppl} for iteration: {iters}')
                    print('\n')
                    
                    val_ppl.append(ppl)
                    val_loss.append(loss)
                    
                                
                    np.save(self.statsfolder + '/' + 'val_PPL_iter'  + '.npy', np.array(val_ppl))
                    
                    
                    np.save(self.statsfolder + '/' + 'train_rewards' + '.npy', np.array(rewards))
                    np.save(self.statsfolder + '/' + 'train_kl' + '.npy', np.array(kl))
                    np.save(self.statsfolder + '/' + 'train_clip_frac' + '.npy', np.array(clip_frac))
                    np.save(self.statsfolder + '/' + 'best_ppl_iteration_value' + '.npy', np.array(iters))
                    


                    #self.initialize_strategy_count()
    
                    if not self.train_single_model:
                        self.model_A.train()
                        self.model_B.train()
                    else:
                        if self.single_model_to_train == 'agent':
                            self.model_A.train()
                #if update_count == 17:
                #    return best_ppl, iters
        return best_ppl, iters

class NegotiationDataset(Dataset):
    def __init__(self, data,next_intent,next_utter, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer.max_len = 1500
        self.next_utter = next_utter
        self.next_intent = next_intent
        # tokenizer weird behavior
        self.turn_ending = [628, 198]


        # tokenizer.encode("\n\n")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        dial_tokens = []
        '''print("l_data:",self.data[index])
        print("l_next_utter:",self.next_utter[index])
        print("l_next_intent:",self.next_intent[index])'''
        
        for i in range(len(self.data[index])):
            sep1 = "\t"
            
            if self.data[index][i][0]=='A':
                sep1 = "next seller response \t"
               # print("$$$$")
            elif self.data[index][i][0]=='B':
                sep1 = "next buyer response \t"
                #print("$$$$")
            else:
                print("### Not Getting ###")
                print(self.data[index][i])
            sep2 = 'with intent: \t'
            
            #print(len(self.data[index]), len(self.next_intent[index]),len(self.next_utter[index]) )
            #print(self.next_intent[index][i],self.next_utter[index][i])
            dial_tokens.append(self.tokenizer.encode(self.data[index][i]) + self.tokenizer.encode(sep1) + self.tokenizer.encode(self.next_utter[index][i]) + self.tokenizer.encode(sep2) + self.tokenizer.encode(self.next_intent[index][i]) + self.tokenizer.encode(' <|end|>') + self.turn_ending)
            #dial_tokens.append(tokenizer.encode(self.data[index][i]) + tokenizer.encode(sep)  + tokenizer.encode(self.next_utter[index][i])+ self.turn_ending)
        role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
        return role_ids, dial_tokens
        
    def collate(self, unpacked_data):
        return unpacked_data
   #r1 
if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("cuda--in--use")
    else:
        print("no cuda available!!!!")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    device = torch.device("cuda" if use_cuda else "cpu")
    print('device_type: ',device)
    trainer = Trainer(modelname='model_v1_nsr_1000',
                      csvfile="dialogue_data_vs1_1.csv",
                      device=torch.device("cuda"),
                      n_epochs=5,
                      batch_size=1,
                      mini_batch=20,
                      train_single_model=False,
                      single_model_to_train = 'agent',
                      num_candidates=3,
                      recompute_log_prob=True,
                      average_sent_loss=True,
                      max_candidate_length=300,
                      human_reward=10,
                      top_p=0.9,
                      temperature=0.8,
                      use_recent_past=True,
                      warmup_steps=10,
                      print_every=1,
                      evaluate_every=500,
                      learning_rate=2e-5,
                      epsilon=0.2,
                      loadModel=False,
                      loadFilenameA="/model_v1/model_v1_A1200.pth",
                      loadFilenameB="/model_v1/model_v1_B1200.pth",
                      pad_token_id=2,
                      seedvalue=10, # 10 should be the seed value since pre trained on the same seed. 
                      use_intent=True,
                      use_prize_gap=True,
                      use_nego_strategy=False,
                      use_surface_similarity=True,
                      gamma1=0.2,
                      gamma2=0.3,
                      gamma3=0.3,
                      gamma4=0.2)

    trainer.train()