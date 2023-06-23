import numpy as np
import torch.nn as nn
from nltk.translate.meteor_score import meteor_score
import nltk
#nltk.download('wordnet')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import functools
import operator
import pdb
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
#from persona_nli import *
import warnings
from get_intent import BertClassifier_buyer,BertClassifier_seller
import numpy as np
from collections import Counter
import math
warnings.filterwarnings("ignore")

model = SentenceTransformer('bert-base-nli-mean-tokens')

def convert_sentences_to_strings(sentences:list, tokenizer):
    str_sentences = []
    for i in sentences:
        str_sentences.append(tokenizer.decode(i.tolist()[0][2:-2])) # Excludeqs the zero shot tokens: {A:, B:} and the End of turn tokens: [628, 198]
    return str_sentences

def normalize(text, nlp):
    sent = ''
    doc = nlp(text)
    for token in doc:
        if not token.is_punct:
            sent += token.lemma_
            sent += ' '
    return sent

def label2idx(pred_intent,role_id):
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
        'acknowledge-acceptance':8,
        'greet-inform_negotiate-price-nochange':9,
        'negotiate-remove-delivery':10,
        'negotiate-price-remove-x':11,
        'avoid_rejection':12
    }
  if role_id==0:
    return buyer_intent[pred_intent.lower()]
  else:
    return seller_intent[pred_intent.lower()]
    


def intent_pred(current_sentence,pred_intent,role_id):
  
  intent_reward = []
  idx = -1
  print("Inside intent_pred_module")
  for i in range(len(current_sentence)):
    indx = current_sentence[i].find('next')
    current_sentence[i] = current_sentence[i][indx:]
  print(current_sentence[0])
  for i in range(len(current_sentence)):
    print('pred_intent',pred_intent[i])
    if pred_intent[i]!='NA':
        
        
        idx = label2idx(pred_intent[i].lower(),role_id)
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
        print('Not taking Intent for: ',current_sentence[i])
        intent_reward.append(0.54)
  return intent_reward

def prize_gap(seller_ip,seller_final):
  res = []
  for i in range(len(seller_ip)):
      res.append(seller_final[i]/seller_ip[i])
  return res

def neg_eval(prize_diff):
  print("****prize_diff:",prize_diff)
  pr_Dif = []
  if prize_diff>=0:
    #return 1/(1 + np.exp(-prize_diff))
    return np.exp(prize_diff)
    
  else:
    return 0
    
 # 1 
def negotiation_result(result,buyer_price,seller_min_price):
  rfactor = 0
  val_ret = []
  for i in range(len(buyer_price)):
    if result == 'Accept':
        rfactor = 1
    else:
        rfactor = -1
    val_ret.append(neg_eval(((buyer_price[i] - seller_min_price[i])/(buyer_price[i]+1))*rfactor))
  return val_ret


def sentence_surface_similarity(curr_sentence,context):
  
  cos_values = []
  sim_val = []
  #print(curr_sentence)
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
  



def calculate_rewards(model_A,
                      current_sentence,
                      num_turn,
                      dial_inputs,
                      length,
                      generated_sentences,
                      source_list,
                      conv_sentences,
                      tokenizer,
                      criterion,
                      buyer_price,
                      seller_min_price,
                      seller_initial_price,
                      pred_intent,
                      final_intent,
                      prize_diff,
                      role_id,
                      use_intent,
                      use_prize_gap,
                      use_nego_strategy,
                      use_surface_similarity,
                      nlp,
                      device,
                      gamma1,
                      gamma2,
                      gamma3,
                      gamma4,
                      agent=False):

    
    scores = {}

    scores['intent'] = []
    scores['negotiation_strategy'] = []
    scores['prize_gap'] = []
    scores['sentence_similarity'] = []
    intent_logit = 0
    deal_scores = 0
    similarity_scores = 0
    strategy_score = 0

    if len(generated_sentences) >= 1:
        
        rewards = np.zeros(len(generated_sentences))
        
        # if (len(source_list) ==2):
            
        if use_intent:
            if pred_intent!='NA':
                intent_logit = intent_pred(generated_sentences,pred_intent,role_id)
            else:
                intent_logit = np.zeros(len(generated_sentences))
            # dial_length = np.array(length)

            intent_logit = np.array(intent_logit)

            # engagingness = 0.75*non_rep+2*dial_length

            rewards += gamma1*(intent_logit)
        else: 
            intent_logit = np.zeros(len(generated_sentences))
        
        if use_prize_gap:
            deal_scores = prize_gap(seller_initial_price,buyer_price)
            print("deal score: ",deal_scores)
            rewards += gamma2*np.array(deal_scores)
        
        if not use_prize_gap:
            deal_scores = np.zeros(len(generated_sentences))


        if use_nego_strategy:
            print("Input to neg strategy")
            print(final_intent, buyer_price,seller_min_price)
            strategy_score = negotiation_result(final_intent, buyer_price,seller_min_price)
            print("strtegy score: ",strategy_score)
            rewards += gamma3*np.array(strategy_score)
        
        if not use_nego_strategy:
            strategy_score = np.zeros(len(generated_sentences))


        if use_surface_similarity:
            if len(conv_sentences)>0:
                similarity_scores = sentence_surface_similarity(generated_sentences,conv_sentences)
            else:
                similarity_scores = np.zeros(len(generated_sentences))
            rewards -= gamma4*np.array(similarity_scores)
        if not use_surface_similarity:
            similarity_scores = np.zeros(len(generated_sentences))
        
        




    else:
        rewards = 0
        
        
    
    
    scores['intent'].extend([intent_logit])
    scores['negotiation_strategy'].extend([strategy_score])
    scores['prize_gap'].extend([deal_scores])
    scores['sentence_similarity'].extend([similarity_scores])

    #print('reward:',rewards)
    

    return list(rewards), scores

def append(generated_list, context_sentence, tokenizer):
    
    if len(generated_list) == 2:
        generated_list.pop(0)
        cntx = tokenizer.decode(context_sentence.tolist()[0][2:]).split('\n')[0]
        generated_list.append(cntx)
    else:
        cntx = tokenizer.decode(context_sentence.tolist()[0][2:]).split('\n')[0]
        generated_list.append(cntx)
    
    return generated_list

def expand_inputs_for_N_candidates(inputs, num_candidates):
    # inputs = inputs[None, ...]
    return inputs.repeat((num_candidates, 1))

def modify_generated_sequence(generated_sequences, generated_log_probs):
    
    final_generated_sequences = []
    final_generated_log_probs = []
    
    for i in range(generated_sequences.shape[0]):
        
        batch_tokens = []
        batch_log_probs = []
        
        for j in range(len(generated_sequences[i])):
            if generated_sequences[i][j] != 628 and generated_sequences[i][j] != -1:
                batch_tokens.append(generated_sequences[i][j])
                batch_log_probs.append(generated_log_probs[i][j])
            elif generated_sequences[i][j] == 628:
                batch_tokens.append(generated_sequences[i][j])
                batch_log_probs.append(generated_log_probs[i][j])
                batch_tokens.append(198)
                break
            else:
                break
        final_generated_sequences.append(torch.tensor(batch_tokens).unsqueeze(0))
        ### BE CAREFUL WHEN USING THIS, SINCE IT DOESN NOT AVERAGES THE LOG PROBS INSTEAD IT JUST TAKES THE SUM.
        final_generated_log_probs.append(torch.tensor(batch_log_probs).sum().item())
    
    return final_generated_sequences, final_generated_log_probs

def top_p_candidates(logits, prob=0.92, filter_value=-float('Inf')):
    
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cum_sum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cum_sum > prob
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter_(1, index=sorted_indices, src=sorted_indices_to_remove.clone())
    logits[indices_to_remove] = filter_value
    
    return logits

def generate_n_candidates(model,
                          inputs,
                          top_p,
                          temperature,
                          num_candidates,
                          max_gen_length,
                          past,
                          device,
                          eos_token_id=628,
                          pad_token_id=198):

    curr_len = inputs.shape[1]
    #print('curr_len',curr_len)
    if inputs.shape[1]>200:

        max_gen_length=300
    else:
        max_gen_length=210
    inputs = expand_inputs_for_N_candidates(inputs, num_candidates)
    inputs_ = inputs
    #print(inputs.shape)
    generated_sequences = torch.ones((inputs.shape[0], max_gen_length), dtype=torch.long) * -1
    #print(generated_sequences.shape)
    generated_sequences[:, 0:inputs.shape[1]] = inputs.cpu()
    
    generated_token_log_prob = torch.zeros((inputs.shape[0], max_gen_length), dtype=torch.float)
    
    unfinished_sequences = inputs.new(inputs.shape[0]).fill_(1) #.cpu()
    
    i = 0
    
    while True:
        if past:
            if past[0][0].shape[-2] > 1024:
                if not torch.all(generated_sequences==-1):
                    final_generated_sequence, final_generated_log_probs = modify_generated_sequence(generated_sequences, generated_token_log_prob)
                    return final_generated_sequence, final_generated_log_probs, past
                else:
                    return None, None
        
        outputs = model(inputs, past)
        logits, past = outputs[0], outputs[1]
        
        next_token_logits = logits[:, -1, :].contiguous() / temperature
        
        if top_p and top_p > 0.0:
            # This returns score after performing softmax function.
            next_token_logits = top_p_candidates(next_token_logits, top_p)
            next_token_log_probs = F.log_softmax(next_token_logits, -1)
            probs = F.softmax(next_token_logits, dim=-1)
            
            next_tokens = torch.multinomial(probs, num_samples=1)
            next_token_log_probs = next_token_log_probs.gather(-1, next_tokens)
            next_tokens = next_tokens.squeeze(1)
            
            if eos_token_id is not None:
                assert pad_token_id is not None # "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            # NOTE: SAVE LOG PROBS AS WELL
            generated_sequences[:, curr_len] = next_tokens.cpu()
            inputs = next_tokens.unsqueeze(1).to(device)
            #inputs_ = torch.cat((inputs_, next_tokens[:, None]), dim=-1)
            
            curr_len = curr_len + 1
            
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
            if unfinished_sequences.max() == 0:
                break
            if curr_len >= max_gen_length:
                break
    
    final_generated_sequences, final_generated_log_probs =  modify_generated_sequence(generated_sequences, generated_token_log_prob)
    
    return final_generated_sequences, final_generated_log_probs

def compute_log_probs(target_token_ids, logits, mask, average_sent_loss=False):
    logits = logits[:, :-1, :].contiguous() # (batch, sequence_length, vocab_size)
    
    target_token_ids = target_token_ids[:, 1:].contiguous() # (batch, sequence_length)
    

    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs, -1, target_token_ids.unsqueeze(-1)).squeeze(-1)
    mask = mask[:, 1:].contiguous()
    
    if average_sent_loss:
        log_probs = (log_probs * mask).sum(-1) / mask.sum(-1)
    else:
        log_probs = (log_probs * mask).sum(-1)
    return {'log_probs': log_probs}

def get_logits_output(logits):
    next_token_logits = logits[:, -1, :].contiguous() / 0.8
    next_token_logits = top_p_candidates(next_token_logits, 0.92)
    next_token_log_probs = F.log_softmax(next_token_logits, -1)
    probs = F.softmax(next_token_logits, dim=-1)
            
    next_tokens = torch.multinomial(probs, num_samples=1)
    next_token_log_probs = next_token_log_probs.gather(-1, next_tokens)
    next_tokens = next_tokens.squeeze(1)
    return next_tokens.cpu()

def get_predicted_intent(sent3,role_id):
    
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
    indx = sent3.find('next')
    sent3 = sent3[indx:]
    sent = sent3.lower()
    sent = sent.split(' ')
    sent = [k.strip() for k in sent]
    sent.reverse()
    #print(sent)
    Flag = False
    if role_id==0:
        key_li = buyer_intent.keys()
    else:
        key_li = seller_intent.keys()
    intent = 'NA'
    key_li = [ele.lower() for ele in key_li]
    #print(key_li)
    for ele in sent:
      for k in key_li:
        #print(k,ele)
        if k in ele:
          #print('yes')
          Flag = True
          el_len = len(ele.split('_'))
          k_len = len(k.split('_'))
          if k_len == el_len:
            Flag = True
            intent = k
            break
          
        
      if Flag == True:
        break
    
    return intent
 




    #print(intent)
   

def ppo_step(model_A,
             model_B,
             buffer_memory,
             device,
             ppo_epsilon,
             num_candidates,
             criterion,
             optimizer,
             dial_inputs,
             role_ids,
             scheduler=None,
             train_single_model=False,
             model_to_train=None,
             average_sent_loss=False,
             use_recent_past=False):

    optimizer.zero_grad()
    
    log_dict = {}
    
    new_log_prob = []
    old_log_prob = []
    
    rewardlist = []
    
    ratios = []
    
    policy_loss = []
    advantages  = []

    if use_recent_past:
        print('USING RECENT PAST')
    else:
        print('NOT USING RECENT PAST')

    if use_recent_past:
        
        batches = buffer_memory.get_batch(shuffle=False)
        
        past = None
        
        i = 0
        
        for idx, batch in enumerate(batches):
            
            action = torch.tensor(batch['action'], device=device).unsqueeze(0)
            #pdb.set_trace()      
            if batch['human_response']:
                
                if idx == 0:
                    logits, past = model_A(action, past, return_dict=False)
                
                if idx > 0 and idx % (num_candidates + 1) == 0:
                    try:
                        past = out
                    except:
                        pass
                    
                    #history_indices = idx // (num_candidates + 1)
                    #history = dial_inputs[history_indices]
                    if i < len(dial_inputs):
                        history = dial_inputs[i]
                    else:
                        continue
                    
                    _, past = model_A(history.to(device), past_key_values=None, return_dict=False)
                    logits, out = model_A(action, past_key_values=past, return_dict=False)
                    
                    i += 2
            else:
                history_indices = idx // (num_candidates + 1)  # {A:(1,2,3,4,5),B, C:(7,8,9,10,11), D, E: (13,14,15,16,17)}
                
                if history_indices == 0:
                    logits, _ = model_A(action, past_key_values=None, return_dict=False)
                else:
                    logits, _ = model_A(action, past_key_values=past, return_dict=False)
            
            new_log_probs = compute_log_probs(target_token_ids=action,
                                              logits=logits,
                                              mask=torch.ones_like(action).to(device),
                                              average_sent_loss=average_sent_loss)['log_probs']

            old_log_probs = torch.tensor(batch['log_prob'], device=device).unsqueeze(0)
            old_log_prob.append(old_log_probs)

            rewards = torch.tensor(batch['reward'], device=device).unsqueeze(0)
            rewardlist.append(batch['reward'])
            advantages.append(rewards)

            new_log_prob.append(new_log_probs)

        if new_log_prob:
            new_log_prob = torch.cat(new_log_prob, dim=-1)
            old_log_prob = torch.cat(old_log_prob, dim=-1)
        
            advantages = torch.cat(advantages, dim=-1)
        
            ratio = (new_log_prob - old_log_prob).exp()
        
            policyloss1 = - advantages * ratio
            policyloss2 = - advantages * ratio.clamp(1 - ppo_epsilon, 1 + ppo_epsilon)
        
            policyloss = torch.min(policyloss1, policyloss2).mean()
        
            policyloss.backward()

            with torch.no_grad():
                log_dict['policy_loss'] = policyloss.item()
                print('Policy Loss: ', log_dict['policy_loss'])
                
                # (r-1) - logr, where r = p(x)/q(x); p(x) = new distribution and q(x) is old distribution
                log_dict['approx_kl'] = torch.mean(((new_log_prob - old_log_prob).exp() - 1)\
                                                - (new_log_prob - old_log_prob)).item()
                #log_dict['approx_kl'] = 0.5 * np.mean(np.power((np.array(new_log_prob) - np.array(old_log_prob)), 2))
                print('approx KL div: ', log_dict['approx_kl'])
                
                log_dict['clip_frac'] = torch.mean((torch.abs(ratio-1) > ppo_epsilon).float()).item()
                print('clip frac: ', log_dict['clip_frac'])
                
                log_dict['reward'] = np.mean(rewardlist)
                print('rewards: ', log_dict['reward'])
        else:
            log_dict['policy_loss'] = 0
            print('Policy Loss: ', log_dict['policy_loss'])
                
            # (r-1) - logr, where r = p(x)/q(x); p(x) = new distribution and q(x) is old distribution
            log_dict['approx_kl'] = 0
            
            #log_dict['approx_kl'] = 0.5 * np.mean(np.power((np.array(new_log_prob) - np.array(old_log_prob)), 2))
            print('approx KL div: ', log_dict['approx_kl']) 

            log_dict['clip_frac'] = 0
            print('clip frac: ', log_dict['clip_frac'])
                
            log_dict['reward'] = 0
            print('rewards: ', log_dict['reward'])
        

    if not train_single_model:
        nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)
        nn.utils.clip_grad_norm_(model_B.parameters(), 1.0)
    else:
        if model_to_train =='agent':
            nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)

    optimizer.step()
    #scheduler.step()

    return log_dict
def slice_input(dial_turn_input):
    #print('yes')
    slice_len = 0
    #print('shape of turn 1: ',dial_turn_input.shape[1])
    for i in range(dial_turn_input.shape[1]):
        if dial_turn_input[0][i] == 197:
            #print("tensor: ",dial_turn_input[0][i])
           # print("*****")
            return slice_len
        else:
            ##print("tensor: ",dial_turn_input[0][i])
            #print("else")
            slice_len +=1
    return slice_len


@torch.no_grad()
def collect_samples(batch,
                    model_A,
                    model_B,
                    top_p,
                    eos_token_id,
                    pad_token_id,
                    max_gen_length,
                    num_candidates,
                    human_reward,
                    use_intent,
                    use_prize_gap,
                    use_nego_strategy,
                    use_surface_similarity,
                    buffer_memory,
                    device,
                    tokenizer,
                    criterion,
                    temperature,
                    use_recent_past,
                    average_sent_loss,
                    nlp,
                    gamma1,
                    gamma2,
                    gamma3,
                    gamma4,
                    train_single_model=True,
                    model_to_train=None,
                    recompute_log_prob=True,
                    fp16=False):

    scores_dict = {}

    scores_dict['intent'] = []
    scores_dict['negotiation_strategy'] = []
    scores_dict['prize_gap'] = []
    scores_dict['sentence_similarity'] = []
    
    print("In training Step")

    role_ids, dialog_tokens = batch
    
    dial_inputs = [torch.LongTensor(item).unsqueeze(0) for item in dialog_tokens]


    pastA = None
    pastB = None
    past = None
    past_ = None
    past_A = None
    past_B = None
    context = None
    cntxt = None

    agent_generated_list, user_generated_list = [], []
    length = np.zeros(num_candidates)
    length = length.tolist()

    conv_sentences = []
    final_conv = len(dial_inputs)-1
    
    
    buyer_price = 0
    prize_diff = 0
    i=0
    t=0
    final_intent = ''
    seller_initial_price = 10000
    init_price=False
    seller_minimum_price=10000
    for num_turn, dialog_turn_inputs in enumerate(dial_inputs):
        print("#### loop i",t)
        t+=1
        assert not np.any(np.isnan(dialog_turn_inputs).cpu().numpy()), 'Inputs Dialog contains Nan value.'
        
        dialog_turn_inputs = dialog_turn_inputs.to(device)
        
        current_sentence = tokenizer.decode(dialog_turn_inputs.tolist()[0][2:]).split('\t')[0]
    
        if i==1 or init_price==False:
            print("init_conv")
            curr_sent_word = current_sentence.split(' ')
            curr_sent_word = [i.strip() for i in curr_sent_word if len(i)>0]
            get_price = False
            for word in curr_sent_word:
                if word == '#price':
                    get_price = True
                    continue
                if get_price == True:
                    word_ = word.replace(".","")
                    word_ = word_[1:-3]
                    if word_.isnumeric():
                        if word[0]=='$':
                            word = word[1:-1]
                        ab = "'23'"
                        #print("word:",word.strip())
                        #print(":",ab)
                    
                        seller_initial_price = int(float(word))
                        init_price=True
                        break
                    else:
                        init_price=False
                        #print("Not number")
                        #print("word:",word.strip())
                        break
        
           
        if init_price==True:           
            seller_minimum_price = 0.7*seller_initial_price
            print("seller_initial_price:",seller_initial_price)
     
        if i == final_conv:
            print("final_conv")
            
            if role_ids[num_turn] == 0:
                final_intent = 'Reject'
                curr_sent_word = current_sentence.split(' ')
                get_price = False
                for word in curr_sent_word:
                    if word == '#price':
                        #print("#price found")
                        get_price = True
                        continue
                    if get_price == True:
                        if word[0]=='$':
                            word = word[1:-1]
                        else:
                            word = word[0:-1]
                        buyer_price = int(float(word))
                        print("Buyer price in role0:",buyer_price)
                        break
            elif role_ids[num_turn]==1:
                
                final_intent = 'Accept'
                
                #final_intent = 'Reject'
                curr_sent_word = current_sentence.split(' ')
                #print(curr_sent_word)
                get_price = False
                for word in curr_sent_word:
                    if word == '#price':
                        #print("#price found")
                        get_price = True
                        continue
                    if get_price == True:
                        if word[0]=='$':
                            word = word[1:-1]
                        else:
                            word = word[0:-1]
                        buyer_price = int(float(word))
                        print("Buyer price in role1:",buyer_price)
                        break
                
            prize_diff = buyer_price - seller_minimum_price
        if not train_single_model:
            if role_ids[num_turn] == 0:
                #pastA = None
                outputs = model_A(dialog_turn_inputs, past, return_dict=False)
                logits = outputs[0]
                index = slice_input(dialog_turn_inputs)
                new_input = dialog_turn_inputs[:,0:index]
                #sent_logit = get_logits_output(logits)
                #print('sent_logits: ',sent_logit)
                print('input: \n')
                print(convert_sentences_to_strings([dialog_turn_inputs], tokenizer)[0])
                #print("tensor input\n")
                #print(dialog_turn_inputs)
                #pred_sentence = convert_sentences_to_strings([sent_logit], tokenizer)[0]
                #pred_intent = get_predicted_intent(pred_sentence)
                mask = torch.ones_like(dialog_turn_inputs).to(device)
                #past = None
                #past_ = None
                generated_sequence, generated_log_probs = generate_n_candidates(model_A,
                                                                new_input, top_p,
                                                                eos_token_id=eos_token_id,
                                                                pad_token_id=pad_token_id,
                                                                num_candidates=num_candidates,
                                                                max_gen_length=max_gen_length,
                                                                temperature=temperature,
                                                                past=past_,
                                                                device=device)
                pred_sentence = convert_sentences_to_strings(generated_sequence, tokenizer)

                #pred_intent = get_predicted_intent(pred_sentence,role_ids[num_turn])
                #pred_intent = 'Accept'
                print('Generated sentnce:\n', pred_sentence[0])
                pred_intent = []
                buyer_p = []
                seller_min_pr = []
                seller_init_price = []
                for sent in pred_sentence:

                    pred_intent.append(get_predicted_intent(sent,role_ids[num_turn]))
                    buyer_p.append(buyer_price)
                    seller_min_pr.append(seller_minimum_price)
                    seller_init_price.append(seller_initial_price)
                log_probs = compute_log_probs(target_token_ids=dialog_turn_inputs,
                                              logits=logits,
                                              mask=mask,
                                              average_sent_loss=average_sent_loss)
                
                buffer_memory.update_buffer(state=dialog_turn_inputs.tolist()[0],
                                            context=context,
                                            action=dialog_turn_inputs.tolist()[0],
                                            action_log_probs=log_probs['log_probs'].item(),
                                            reward=human_reward,
                                            agent=True,
                                            human_response=True)
                if not use_recent_past:
                    '''In this case, first we generate sentence using the entire past. And then we update the past with
                    the current utterance.'''
                
                    generated_sequence, generated_log_probs  = generate_n_candidates(model_A,
                                                                                     new_input,
                                                                                     top_p,
                                                                                     eos_token_id=eos_token_id,
                                                                                     pad_token_id=pad_token_id,
                                                                                     num_candidates=num_candidates,
                                                                                     max_gen_length=max_gen_length,
                                                                                     temperature=temperature,
                                                                                     past=past_,
                                                                                     device=device)
                    pred_sentence = convert_sentences_to_strings(generated_sequence, tokenizer)[0]
                    pred_intent = get_predicted_intent(pred_sentence,role_ids[num_turn])
                    pred_intent = []
                    buyer_p = []
                    seller_min_pr = []
                    seller_init_price = []
                    for sent in pred_sentence:

                        pred_intent.append(get_predicted_intent(sent,role_ids[num_turn]))
                        buyer_p.append(buyer_price)
                        seller_min_pr.append(seller_minimum_price)
                        seller_init_price.append(seller_initial_price)
                    #pred_intent = 'Accept'
                    
                    output = model_A(expand_inputs_for_N_candidates(dialog_turn_inputs,num_candidates),
                                     past_,
                                     return_dict=False)

                    past_ = output[1]
            else:
                #pastB = None
                #past = None
                outputs = model_B(dialog_turn_inputs, past, return_dict=False)
                logits = outputs[0]
                sent_logit = get_logits_output(logits)
                index = slice_input(dialog_turn_inputs)
                new_input = dialog_turn_inputs[:,0:index]
                #pred_sentence = convert_sentences_to_strings([sent_logit], tokenizer)[0]
                #pred_intent = get_predicted_intent(pred_sentence)
                mask = torch.ones_like(dialog_turn_inputs).to(device)
                #past = None
                #print('input: \n')
                #print(convert_sentences_to_strings([dialog_turn_inputs], tokenizer)[0])
                #past_ = None
                generated_sequence, generated_log_probs = generate_n_candidates(model_B,
                                                                new_input, top_p,
                                                                eos_token_id=eos_token_id,
                                                                pad_token_id=pad_token_id,
                                                                num_candidates=num_candidates,
                                                                max_gen_length=max_gen_length,
                                                                temperature=temperature,
                                                                past=past_,
                                                                device=device)
                pred_sentence = convert_sentences_to_strings(generated_sequence, tokenizer)
                pred_intent = []
                buyer_p = []
                seller_min_pr = []
                seller_init_price = []
                for sent in pred_sentence:

                    pred_intent.append(get_predicted_intent(sent,role_ids[num_turn]))
                    buyer_p.append(buyer_price)
                    seller_min_pr.append(seller_minimum_price)
                    seller_init_price.append(seller_initial_price)

                #pred_intent = 'Accept'
                #print('Generated sentece:', pred_sentence)
                log_probs = compute_log_probs(target_token_ids=dialog_turn_inputs,
                                              logits=logits,
                                              mask=mask,
                                              average_sent_loss=average_sent_loss)
                
                buffer_memory.update_buffer(state=dialog_turn_inputs.tolist()[0],
                                            context=context,
                                            action=dialog_turn_inputs.tolist()[0],
                                            action_log_probs=log_probs['log_probs'].item(),
                                            reward=human_reward,
                                            agent=True,
                                            human_response=True)
                
                if not use_recent_past:
                    '''In this case, first we generate sentence using the entire past. And then we update the past with
                    the current utterance.'''
                    #past_ = None
                    generated_sequence, generated_log_probs  = generate_n_candidates(model_B,
                                                                                     new_input,
                                                                                     top_p,
                                                                                     eos_token_id=eos_token_id,
                                                                                     pad_token_id=pad_token_id,
                                                                                     num_candidates=num_candidates,
                                                                                     max_gen_length=max_gen_length,
                                                                                     temperature=temperature,
                                                                                     past=past_,
                                                                                     device=device)
                    pred_sentence = convert_sentences_to_strings(generated_sequence, tokenizer)
                    #print('Generated sentece:', pred_sentence)
                    #pred_intent = get_predicted_intent(pred_sentence,role_ids[num_turn])
                   # pred_intent = 'Accept'
                    pred_intent = []
                    buyer_p = []
                    seller_min_pr = []
                    seller_init_price = []
                    for sent in pred_sentence:

                        pred_intent.append(get_predicted_intent(sent,role_ids[num_turn]))
                        buyer_p.append(buyer_price)
                        seller_min_pr.append(seller_minimum_price)
                        seller_init_price.append(seller_initial_price)
                    output = model_B(expand_inputs_for_N_candidates(dialog_turn_inputs,num_candidates),
                                     past_,
                                     return_dict=False)

                    past_ = output[1]

        elif model_to_train == 'agent':
            
            
            if role_ids[num_turn] == 0:
                
                '''if use_recent_past:
                    if cntxt is not None:
                        past = prepare_inputs(cntxt, model_A)
                    else:
                        past = None'''
                
                #dial_turn_str = convert_sentences_to_strings([dialog_turn_inputs], tokenizer)[0]

                outputs = model_A(dialog_turn_inputs, pastA, return_dict=False)
                logits = outputs[0]
                #sent_logit = get_logits_output(logits)
                #pred_sentence = convert_sentences_to_strings([sent_logit], tokenizer)[0]
                #pred_intent = get_predicted_intent(pred_sentence)
                mask = torch.ones_like(dialog_turn_inputs).to(device)
                index = slice_input(dialog_turn_inputs)
                new_input = dialog_turn_inputs[:,0:index]
                input_act = convert_sentences_to_strings([dialog_turn_inputs], tokenizer)[0]

                
                log_probs = compute_log_probs(target_token_ids=dialog_turn_inputs,
                                              logits=logits,
                                              mask=mask,
                                              average_sent_loss=average_sent_loss)
                
                buffer_memory.update_buffer(state=dialog_turn_inputs.tolist()[0],
                                            context=context,
                                            action=dialog_turn_inputs.tolist()[0],
                                            action_log_probs=log_probs['log_probs'].item(),
                                            reward=human_reward,
                                            agent=True,
                                            human_response=True)
                
                if not use_recent_past:
                    '''In this case, first we generate sentence using the entire past. And then we update the past with
                    the current utterance.'''
                
                    generated_sequence, generated_log_probs  = generate_n_candidates(model_A,
                                                                                     new_input,
                                                                                     top_p,
                                                                                     eos_token_id=eos_token_id,
                                                                                     pad_token_id=pad_token_id,
                                                                                     num_candidates=num_candidates,
                                                                                     max_gen_length=max_gen_length,
                                                                                     temperature=temperature,
                                                                                     past=past_,
                                                                                     device=device)
                    pred_sentence = convert_sentences_to_strings(generated_sequence, tokenizer)[0]
                    #pred_intent = get_predicted_intent(pred_sentence)
                    pred_intent = 'Accept'
                    output = model_A(expand_inputs_for_N_candidates(dialog_turn_inputs,num_candidates),
                                     past_,
                                     return_dict=False)
                    '''Here first we calculate the past based on the context sentence and then we generate candidates.'''
                    '''if cntxt is not None:
                        past_ = prepare_inputs(expand_inputs_for_N_candidates(cntxt, num_candidates), model_A)
                    else:
                        past_ = None'''

                    generated_sequence, generated_log_probs = generate_n_candidates(model_A,
                                                                                    new_input, top_p,
                                                                                    eos_token_id=eos_token_id,
                                                                                    pad_token_id=pad_token_id,
                                                                                    num_candidates=num_candidates,
                                                                                    max_gen_length=max_gen_length,
                                                                                    temperature=temperature,
                                                                                    past=past_,
                                                                                    device=device)
                    pred_sentence = convert_sentences_to_strings(generated_sequence, tokenizer)[0]
                    #print('Generated sentece:', pred_sentence)
                    #pred_intent = get_predicted_intent(pred_sentence)
                    #print(pred_intent)
                    pred_intent = 'Accept'
        gen_sent = convert_sentences_to_strings(generated_sequence, tokenizer)
        #print(gen_sent)
        agent_generated_list = append(agent_generated_list, dialog_turn_inputs, tokenizer)

        
        
        use_prize_gap =     False
        use_nego_strategy = False



        reward, scores = calculate_rewards(current_sentence=current_sentence,
                                            num_turn=num_turn,
                                            dial_inputs=dial_inputs,
                                            generated_sentences= pred_sentence,
                                            length=length,
                                            source_list=agent_generated_list,
                                            tokenizer=tokenizer,
                                            criterion=criterion,
                                            agent=True,
                                            role_id=role_ids[num_turn],
                                            prize_diff=prize_diff,
                                            buyer_price=buyer_p,
                                            seller_min_price=seller_min_pr,
                                            seller_initial_price = seller_init_price,
                                            final_intent=final_intent,
                                            conv_sentences=conv_sentences,
                                            use_intent=use_intent,
                                            use_prize_gap=use_prize_gap,
                                            use_nego_strategy=use_nego_strategy,
                                            use_surface_similarity=use_surface_similarity,
                                            pred_intent=pred_intent,
                                            nlp=nlp,
                                            device=device,
                                            gamma1=gamma1,
                                            gamma2=gamma2,
                                            gamma3=gamma3,
                                            gamma4=gamma4,
                                            model_A=model_A)

        print('intent:',scores['intent'])
        print('negotiation_strategy',scores['negotiation_strategy'])
        print('prize_gap:',scores['prize_gap'])
        print('sentence_similarity:',scores['sentence_similarity'])
        scores_dict['intent'].extend(scores['intent'])
        scores_dict['negotiation_strategy'].extend(scores['negotiation_strategy'])
        scores_dict['prize_gap'].extend(scores['prize_gap'])
        scores_dict['sentence_similarity'].extend(scores['sentence_similarity'])
        
        conv_sentences.append(current_sentence)
        if recompute_log_prob:

            for j in range(len(generated_sequence)):
                
                # NOTE: STILL USING THE PAST FROM PREVIOUS UTTERANCE, SINCE WE DO NOT NEED PAST FROM
                #       CONTAINING CURRENT UTTERANCE for GENERATED CANDIDATES
                if role_ids[num_turn] == 0:
                    output = model_A(generated_sequence[j].to(device), past_key_values=past, return_dict=False)
                    logits = output[0]
                    
                    log_probs = compute_log_probs(target_token_ids=generated_sequence[j].to(device),
                                                logits=logits,
                                                mask=torch.ones_like(generated_sequence[j]).to(device),
                                                average_sent_loss=average_sent_loss)['log_probs'].item()
                    
                    buffer_memory.update_buffer(state=dialog_turn_inputs.tolist()[0],
                                                context=context,
                                                action= generated_sequence[j].tolist()[0],
                                                action_log_probs=log_probs,
                                                reward=reward[j],
                                                agent=True,
                                                human_response=False)
                else:
                    output = model_B(generated_sequence[j].to(device), past_key_values=past, return_dict=False)
                    logits = output[0]
                    
                    log_probs = compute_log_probs(target_token_ids=generated_sequence[j].to(device),
                                                logits=logits,
                                                mask=torch.ones_like(generated_sequence[j]).to(device),
                                                average_sent_loss=average_sent_loss)['log_probs'].item()
                    
                    buffer_memory.update_buffer(state=dialog_turn_inputs.tolist()[0],
                                                context=context,
                                                action= generated_sequence[j].tolist()[0],
                                                action_log_probs=log_probs,
                                                reward=reward[j],
                                                agent=True,
                                                human_response=False)

        else:
            for k in range(len(generated_sequence)):
                buffer_memory.update_buffer(state=dialog_turn_inputs.tolis()[0],
                                            action=generated_sequence[k].tolist()[0],
                                            action_log_probs=generated_log_probs[k],
                                            agent=True,
                                            human_response=False)
        if role_ids[num_turn] == 0:
            past = outputs[1]
            #past = None
            outputs = model_A(expand_inputs_for_N_candidates(dialog_turn_inputs, num_candidates), past_, return_dict=False)
            past_ = outputs[1]
            #past_ = None
            #Q
        else:
            past = outputs[1]
            #past = None
            #past = outputs[1]
            outputs = model_B(expand_inputs_for_N_candidates(dialog_turn_inputs, num_candidates), past_, return_dict=False)
            past_ = outputs[1]
            #past_ = None

    
        
        context = dialog_turn_inputs.tolist()[0]
        cntxt = dialog_turn_inputs
        
        i=i+1

    return dial_inputs, role_ids, scores_dict, #candidate_dict

def get_past(batches, model, device):
    
    states = torch.cat(batches, dim=-1).to(device)
    outputs = model(states, past_key_values=None, return_dict=False)
    
    return outputs[1]

def prepare_inputs_for_model(batches, model, num_candidates, device):
    
    states = get_history_utterances(batches, num_candidates)
    states = torch.cat(states, dim=1, device=device)
    outputs = model(states, past_key_values=None, return_dict=False)
    
    return outputs[1]

def get_history_utterances(batches, num_candidates):
    states = []
    for i in range(0, len(batches), num_candidates+1):
        states.append(i)
    return states

def get_recursive_past(dial_inputs, role_ids, model_A, model_B, device):
    '''
    Uses both models alternatively to calculate pasts.
    Used in case of training only the agent.
    '''
    past = None
    for num_turn, utter in enumerate(dial_inputs):
        if role_ids[num_turn] == 0:
            _, past = model_A(utter.to(device), past_key_values=past, return_dict=False)
        else:
            _, past = model_B(utter.to(device), past_key_values=past, return_dict=False)
    return past