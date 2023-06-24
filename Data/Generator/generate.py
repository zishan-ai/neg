from numpy.random import choice
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from Dialogue_flow_generation.e_commerce_dialogue_flow import get_random_flow

print('loading model...')

model = AutoModelForCausalLM.from_pretrained("path",
                                             local_files_only=True).cuda()  # If GPTJ is locally available, else load from huggingface
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")


print('loading data...')


def placeholder():
    return ""


def traverse(o, tree_types=(list, tuple)):
    if isinstance(o, tree_types):
        for value in o:
            for subvalue in traverse(value, tree_types):
                yield subvalue
    else:
        yield o


def parse_flow_temp(flow, data):
    mapping = {}
    for cnt1, item in enumerate(flow[:-2]):

        product = None
        for i in range(4, 7):
            for cnt2, entity in enumerate(list(traverse(item[i]))):

                if entity == '--' or entity == 'NA':
                    continue

                if entity not in mapping:

                    if product == None:
                        product = entity

                    if entity in data[product]['accessories']:

                        random_select = random.choice(
                            list(data[product]['accessories'][entity].items()))

                        to_str = "A " + entity + " called " + random_select[
                            0] + "." + random_select[1]

                        mapping[entity] = to_str

                    else:

                        random_select = random.choice(
                            list(data[product]['description'].items()))

                        to_str = "A " + product + " called " + random_select[
                            0] + "." + random_select[1]

                        mapping[entity] = to_str

    for item in flow[:-2]:
        if item[3] != 'NA' and item[3] != "--" and item[3] in mapping:
            item[3] = mapping[item[3]]

        if len(item[4]) > 0:
            if item[4][0] in mapping:
                item[4][0] = mapping[item[4][0]]

        if len(item[4]) > 1:
            item[4][1] = list(item[4][1])
            for cnt, x in enumerate(item[4][1]):
                if item[4][1][cnt] in mapping:
                    item[4][1][cnt] = mapping[item[4][1][cnt]]

            item[4][1] = tuple(item[4][1])

    for i in range(1, len(flow) - 1):

        if flow[i - 1][1] == 'Negotiate-Remove-X' or flow[
                i - 1][1] == 'Negotiate-Add-X':
            flow[i][1] = 'tell_price'

        if flow[i - 1][1] == 'Negotiate-Remove-X_Negotiate-Price-Decrease':
            flow[i][1] = 'Negotiate-Price-Remove-X'

    return flow


def call_function_for_string(s, turn, product_features, seller_initial,
                             seller_final, customer_initial, customer_final,
                             customer_utterance, seller_utterance,
                             clarification_feature, removal, addition):
    if s == 'NA':
        return placeholder()
    elif s == 'Negotiate-Remove-delivery':
        if turn % 2 == 1:
            return negotiate_remove_delivery(product_features=product_features,
                                             seller_price=seller_initial)
        else:
            return tell_price(sellers_final=seller_final,
                              customers_utterance=customer_utterance)

    elif s == 'Ask_Clarification-Y':
        return ask_clarification(product_feature=product_features,
                                 seller_price=seller_initial,
                                 clarification_feature=clarification_feature)
    elif s == 'Greet-Inform_Negotiate-Price-Increase':
        return greet_inform_negotiate_increase(
            product_feature=product_features,
            seller_initial=seller_initial,
            seller_final=seller_final,
            customers_utterance=customer_utterance)
    elif s == 'Negotiate-Price-Remove-X':
        return negotiate_price_remove_v2(product_features=product_features,
                                         customer_utterance=customer_utterance,
                                         seller_final=seller_final)

    elif s == 'Accept':
        if turn % 2 == 1:
            return accept_customer(product_feature=product_features,
                                   seller_initial=seller_initial,
                                   seller_utterance=seller_utterance)
        else:
            return accept_seller(product_feature=product_features,
                                 customer_initial=customer_initial,
                                 customer_utterance=customer_utterance)
    elif s == 'Greet-Ask_Negotiate-Price-Decrease':
        return greet_ask_negotiate_decrease(product_feature=product_features,
                                            seller_price=seller_initial,
                                            customer_price=customer_final)
    elif s == 'Reject':
        if turn % 2 == 1:
            return reject_customer(product_feature=product_features,
                                   seller_initial=seller_initial,
                                   customer_final=customer_final,
                                   seller_utterance=seller_utterance)
        else:
            return reject_seller(product_feature=product_features,
                                 seller_intial=seller_initial,
                                 seller_final=seller_final,
                                 customer_utterance=customer_utterance)
    elif s == 'Negotiate-Price-Decrease':
        return negotiate_decrease(product_features, customer_initial,
                                  customer_final, seller_utterance)
    elif s == 'Negotiate-Add-X':
        return negotiate_add(product_features=product_features,
                             addition=addition)
    elif s == 'Negotiate-Price-NoChange':
        if turn % 2 == 1:
            return negotiate_noChange_buyer(seller_price=seller_initial,
                                            customer_price=customer_final)
        else:
            return negotiate_noChange_seller(
                product_feature=product_features,
                seller_initial=seller_initial,
                customer_utterance=customer_utterance)
    elif s == 'avoid_rejection':
        return avoid_rejection(customer_utterance=customer_utterance)
    elif s == 'Negotiate-Price-Increase':
        return negotiate_increase(product_features, seller_initial,
                                  seller_final, customer_utterance)
    elif s == 'Greet-Ask':
        return greet_ask(product_features)
    elif s == 'Greet-Inform':
        return greet_inform(product_features=product_features,
                            customer_utterance=customer_utterance,
                            seller_final=seller_final)
    elif s == 'Ask_Price':
        if turn % 2 == 1:
            return ask_price(product_features=product_features)
        else:
            return tell_price(sellers_final=seller_final,
                              customers_utterance=customer_utterance)
    elif s == 'tell_price':
        return tell_price(sellers_final=seller_final,
                          customers_utterance=customer_utterance)
    elif s == 'Provide_Clarification-Y':
        return resolve_clarification(product_features, customer_utterance)
    elif s == 'Negotiate-Remove-X_Negotiate-Price-Decrease':
        return negotiate_remove_decrease_price(product_features,
                                               addition,
                                               customer_final,
                                               seller_price=seller_initial)
    elif s == 'Greet-Inform_Negotiate-Price-NoChange':
        return greet_inform_noChange(product_features,
                                     seller_initial=seller_initial,
                                     customers_utterance=customer_utterance)
    elif s == 'Acknowledge acceptance':
        if turn % 2 == 1:
            return ack_accept_customer(seller_utterance=seller_utterance)
        else:
            return ack_accept_seller(customer_utterance=customer_utterance)

    elif s == 'Negotiate-Remove-X':
        return negotiate_remove(product_features=product_features,
                                removal=removal,
                                seller_price=seller_initial)
    else:
        return placeholder()


f = open("convos.txt", "a")


def gen_utterance(prompt_, temperature, max_len):

    input_ids = tokenizer(prompt_, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids.cuda(),
        do_sample=True,
        temperature=temperature,
        max_new_tokens=150,
        # max_length=max_len,
        pad_token_id=tokenizer.eos_token_id)

    gen_text = tokenizer.batch_decode(gen_tokens)
    # f.write(gen_text[0][len(prompt_):].split("\n\n")[0] + "\n")
    f.write(str(gen_text[0][len(prompt_):].split("\n\n")) + "\n")
    f.write("####\n")

    # return gen_text[0][len(prompt_):].split("\n\n")[0]
    return str(gen_text[0][len(prompt_):].split("\n\n"))


def greet_inform(product_features,
                 customer_utterance,
                 seller_final,
                 temperature=0.75,
                 max_len=950):

    prompt_ = (
        "A seller is selling a laptop called Dell X8 it has 16 GB ram, 11-inch screen and Intel i7 processor.\n"
        '''The customer asks, "Hi, I'm interested in buying the Dell X8 laptop you have listed. How much can I get it for, are there discounts available?"'''
        "The seller will sell the product for $4000, he replies to the customer by saying,\n"
        "<start> Hello! The laptop is being sold for $4000, there aren't any discounts available as of now but I'm sure that it is worth the price!\n\n"
        "A seller is selling a headphone called Beats Solo 3, it has noise cancellation, is over the ear, and has soft ear cushions.\n"
        '''The customer asks, "Hello, I am interested in buying your Beats Solo 3 headphones, how much can I get them for?"'''
        "The seller will sell the product for $200, he replies to the customer by saying,\n"
        "<start> Greetings! Thank you for your interest, the headphones cost $200.\n\n"
        "A seller is selling an iPhone 8, it has 128 GB of memory, 12 MP camera and LTE, and it comes with a 2 year warranty.\n"
        '''The customer asks, "Hi, I'm interested in buying an iPhone 8. How much can I get it for?"'''
        "The seller will sell the product for $500, he replies to the customer by saying,\n"
        "<start> Hey! This iPhone 8 is being sold for $500, it comes with a two year warranty.\n\n"
        "A seller is selling a car called Toyota Camry, it has 50,000 miles on it, a sunroof, and leather seats.\n"
        '''The customer asks, "Hello, I am interested in buying your Toyota Camry. Can you give me more details about the car and its price?"'''
        "The seller will sell the car for $18,000, he replies to the customer by saying,\n"
        "<start> Hello! The Toyota Camry has 50,000 miles on it, it has a sunroof and leather seats. The car is being sold for $18,000.\n\n"
    )

    prompt_ = prompt_ + f"A seller is selling a {product_features}.\n" + f'''The customer asks, "{customer_utterance}".\n''' + \
        f"The seller will sell the car for ${seller_final}, he replies to the customer by saying,\n"

    return gen_utterance(prompt_, temperature, max_len)


def greet_ask_negotiate_decrease(product_feature,
                                 seller_price,
                                 customer_price,
                                 temperature=0.82,
                                 max_len=800):

    prompt_ = (
        "A customer wants to buy a product online from a seller.\n"
        "The customer sees a laptop called Dell X8 it has 16 GB ram, 11-inch screen and Intel i7 processor. The customer wants to ask the seller to sell it for $500.\n"
        "The customer begins the conversation with the seller by saying\n"
        "<start> Hi, I'm interested in buying the Dell X8 laptop you have listed. Can I get it for $500.\n\n"
        "A customer wants to buy a product online from a seller.\n"
        "The customer sees a headphone called Beats Solo 3, it has noise cancellation, is over the ear, and has soft ear cushions. The customer wants to ask the seller to sell it for $100.\n"
        "The customer begins the conversation with the seller by saying\n"
        "<start> Hello, I am interested in buying your Beats Solo 3 headphones, but I was hoping to get a better price. Is there anything you can do to lower the price to $100?\n\n"
        "A customer wants to buy a product online from a seller.\n"
        "The customer sees a men's watch. The customer wants to ask the seller to sell it for $15.\n"
        "The customer begins the conversation with the seller by saying\n"
        "<start> Hello, can I buy your watch for $15?\n\n"
        "A customer wants to buy a product online from a seller.\n")

    inform_feature = "The customer sees a " + product_feature + \
        ". The customer wants to ask the seller to sell it for $" + customer_price + ".\n"

    prompt_ = prompt_ + inform_feature + \
        "The customer begins the conversation with the seller by saying\n"

    return gen_utterance(prompt_, temperature, max_len)


def greet_ask(product_features, temperature=0.8, max_len=850):

    prompt_ = (
        "A customer wants to buy a product online from a seller.\n"
        "The customer sees a laptop called Dell X8 it has 16 GB ram, 11-inch screen and Intel i7 processor.\n"
        "The customer begins the conversation with the seller by saying\n"
        "<start> Hi, I'm interested in buying the Dell X8 laptop you have listed. How much can I get it for, are there discounts available? \n\n"
        "A customer wants to buy a product online from a seller.\n"
        "The customer sees a headphone called Beats Solo 3, it has noise cancellation, is over the ear, and has soft ear cushions.\n"
        "The customer begins the conversation with the seller by saying\n"
        "<start> Hello, I am interested in buying your Beats Solo 3 headphones, how much can I get them for?\n\n"
        "A customer wants to buy a product online from a seller.\n"
        "The customer sees a men's watch.\n"
        "The customer begins the conversation with the seller by saying\n"
        "<start> Hello, how much are you selling the watch for?\n\n"
        "A customer wants to buy a product online from a seller.\n"
        "The customer sees a book called 'The Power of Now' by Eckhart Tolle, it has a paperback binding, 236 pages, and is categorized under the self-help genre.\n"
        "The customer begins the conversation with the seller by saying\n"
        "<start> Hi there, I'm interested in purchasing 'The Power of Now' book by Eckhart Tolle that you have listed. Can you tell me the price?\n\n"
        "A customer wants to buy a product online from a seller.\n")

    prompt_ = prompt_ + "The customer sees a " + product_features + ".\n" + \
        "The customer begins the conversation with the seller by saying\n"

    return gen_utterance(prompt_, temperature, max_len)


def greet_inform_negotiate_increase(product_feature,
                                    seller_initial,
                                    seller_final,
                                    customers_utterance,
                                    temperature=0.7,
                                    max_len=1000):

    prompt_ = (
        "A seller is selling a men's watch. The product is being sold online.\n"
        '''The customer asks, "Hello, can I buy your watch for $15?"\n'''
        "The seller is willing to go down to $25, the seller informs this to the customer by saying: \n"
        "<start> Hello! $15 seems too low for me, the lowest I'm willing to go is $25. Is that okay?\n\n"
        "A seller is selling a laptop called Dell X8, it has 16 GB ram, 11-inch screen and Intel i7 processor. The product is being sold online.\n"
        '''The customer asks, "Hi, I'm interested in buying the laptop that you have listed for my son. I was hoping to get it for $500."\n'''
        "The seller is willing to go down to $770, the seller informs this to the customer by saying: \n"
        "<start> Hello! I'm sure your son would love the laptop, but I'm sorry the lowest I can go is $770. I'm sure you can understand that I need to make a profit on this item, so $500 is not an option. Let me know if you'd like to purchase it at $770.\n\n"
        "A seller is selling a headphone called Beats Solo 3, it has noise cancellation, is over the ear, and has soft ear cushions. The product is being sold online.\n"
        '''The customer asks, "Hello, I am interested in buying your Beats Solo 3 headphones, but I was hoping to get a better price than $149. Is there anything you can do to lower the price to $100?"\n'''
        "The seller is willing to go down to $130, the seller informs this to the customer by saying: \n"
        "<start> Hello there! Thank you for your interest in the Beats Solo 3 headphones. I'm sorry, I am not able to go down to $100, but I am willing to lower the price to $130. If you are interested, please let me know.\n\n"
        "A seller is selling a Blue diamond pendant with sapphire crystal and silver frame. The product is being sold online.\n"
        '''The customer asks,"Hello, I was thinking I'd find a nice pendant for my wife that she'd really love. Does this one look good, and is there anything I can do to get it for $100?"\n'''
        "The seller is willing to go down to $190, the seller informs this to the customer by saying: \n"
        "<start> Hello there! This pendant is beautiful and I'm sure your wife will love it. I'm sorry, but I'm not able to go down to $100. If you are interested in purchasing it for $190, please let me know.\n\n"
    )

    inform_feature = "A seller is selling a " + \
        product_feature + " The product is being sold online.\n"
    prompt_ = prompt_ + inform_feature + f'''The customer asks,"{customers_utterance}"\n''' + \
        "The seller is willing to go down to $" + seller_final + \
        "\n" + "the seller informs this to the customer by saying\n"

    return gen_utterance(prompt_, temperature, max_len)


def negotiate_decrease(product_feature,
                       customer_initial,
                       customer_final,
                       seller_utterance,
                       temperature=0.6,
                       max_len=1000):

    prompt_ = (
        "A customer is negotiating with a seller for a laptop called Dell X8, it has 16 GB ram, 11-inch screen and Intel i7 processor.\n"
        '''The seller tells "I'm sorry, but the lowest I can go is $770. I'm sure you can understand that I need to make a profit on this item, so $500 is not an option. Let me know if you'd like to purchase it at $770."\n'''
        "The highest price that the customer can afford is $570. The customer replies by saying.\n"
        "<start> I appreciate that you need to make a profit on this item, but unfortunately, $770 is above my budget for a laptop. I was ideally hoping to purchase the Dell X8 for 500, but I'm willing to negotiate up to $570 if necessary. Is there any way you could lower the price to meet me somewhere in the middle?\n\n"
        "A customer is negotiating with a seller for a headphone called Beats Solo 3, it has noise cancellation, is over the ear, and has soft ear cushions.\n"
        '''The seller tells "Hello there! Thank you for your interest in the Beats Solo 3 headphones. I'm sorry, I am not able to go down to $100, but I am willing to lower the price to $130. If you are interested, please let me know."\n'''
        "The highest price that the customer can afford is $110. The customer replies by saying.\n"
        "<start> My budget is $110. Can you give a discount and sell it for $110? Let me know if that's possible.\n\n"
        "A customer is negotiating with a seller for a men's watch.\n"
        '''The seller tells, "I'm sorry, but the lowest I'm willing to go is $25. Is that okay?"\n'''
        "The highest price that the customer can afford is $20. The customer replies by saying.\n"
        "<start> Sorry, $25 is above my budget. Can you meet me in the middle and sell for $20?\n\n"
        "A customer is negotiating with a seller for a blue diamond necklace.\n"
        '''The seller tells, "I'm sorry, but the lowest I'm willing to go is $105. Is that okay?"\n'''
        "The highest price that the customer can afford is $90. The customer replies by saying.\n"
        "<start> Sorry, $105 is still above my budget, how about $90?\n\n")

    inform_feature = "A customer is negotiating with a seller for a " + \
        product_feature + ".\n"
    seller_utt = f'''The seller tells "{seller_utterance}"\n'''

    prompt_ = prompt_ + inform_feature + seller_utt + \
        f"The highest price that the customer can afford is ${customer_final}. " + \
        "The customer replies by saying.\n"

    return gen_utterance(prompt_, temperature, max_len)


def negotiate_increase(product_feature,
                       seller_initial,
                       seller_final,
                       customer_utterance,
                       temperature=0.5,
                       max_len=1150):

    prompt_ = (
        "A seller is negotiating with a customer for a laptop called Dell X8, it has 16 GB ram, 11-inch screen and Intel i7 processor.\n"
        '''The customer tells "I appreciate that you need to make a profit on this item, but unfortunately, $770 is above my budget for a laptop. I'm willing to negotiate up to $570 if necessary. Is there any way you could lower the price to meet me somewhere in the middle?"\n'''
        "The lowest that the seller will go is $720. The seller replies by saying.\n"
        "<start> While I appreciate your offer, I cannot go as low as $570. However, I am willing to sell it to you for $720, which is the best price I can offer. This laptop has high-end features such as a 16 GB RAM, 11-inch screen and an Intel i7 processor, and I'm confident it will meet your needs. Please let me know if you would like to purchase it at this price. \n\n"
        "A seller is negotiating with a customer for a headphone called Beats Solo 3, it has noise cancellation, is over the ear, and has soft ear cushions.\n"
        '''The customer tells "My budget is $110. Can you give a discount and sell it for $110? Let me know if that's possible."\n'''
        "The lowest that the seller will go is $120. The seller replies by saying.\n"
        "<start> I understand your budget of $110 and am willing to meet you in the middle at $120. Unfortunately, I cannot go lower than that. I hope we can still come to a mutually beneficial agreement.\n\n"
        "A seller is negotiating with a customer for a men's watch.\n"
        '''The customer tells, "Sorry, $25 is above my budget. Can you meet me in the middle and sell for $20?"\n'''
        "The lowest that the seller will go is $22. The seller replies by saying.\n"
        "<start> I understand that $25 is above your budget, but the best price I can offer for this watch is $22. It is a high-quality product that is worth the investment. Shall we shake on this?\n\n"
        "A seller is negotiating with a customer for a blue diamond necklace.\n"
        '''The customer tells, "Sorry, $125 is still above my budget, how about $90?"\n'''
        "The lowest that the seller will go is $100. The seller replies by saying.\n"
        "<start> I'm sorry but I will not be able to go that low. The necklace is pretty valuable and I'm sure that you will be happy with it. I'm willing to reduce the price to $100. Are you interested?\n\n"
        "A seller is negotiating with a customer for a wooden table.\n"
        '''The customer tells, "Can we negotiate the price, how about $20?"\n'''
        "The lowest that the seller will go is $40. The seller replies by saying.\n"
        "<start> I'm afraid $20 will be a little too low for me. However, I'm willing to reduce the price to $40. Are you interested?\n\n"
    )

    inform_feature = "A seller is negotiating with a customer for a " + \
        product_feature + ".\n"
    customer_utt = f'''The customer tells "{customer_utterance}"\n'''

    prompt_ = prompt_ + inform_feature + customer_utt + \
        f"The lowest the seller will go is ${seller_final}. " + \
        "The seller replies by saying.\n"

    return gen_utterance(prompt_, temperature, max_len)


def ask_clarification(product_feature,
                      seller_price,
                      clarification_feature,
                      temperature=0.8,
                      max_len=700):

    prompt_ = (
        "A customer is in a conversation with a seller about a laptop called Dell X8 it has 16 GB ram, 11-inch screen and Intel i7 processor. The seller is selling it for $800.\n"
        "The customer wants to clarify about its RAM. The customer asks\n"
        "Is the RAM of the Dell X8 laptop 16 GB as advertised?\n\n"
        "The customer is in a conversation with a seller about a headphone called Beats Solo 3, it has noise cancellation, is over the ear, and has soft ear cushions. The seller is selling it at $149.\n"
        "The customer wants to clarify about its audio quality. The customer asks\n"
        "Can you tell me more about the audio quality of the Beats Solo 3 headphones? I'm interested in purchasing them, but I want to make sure they have good sound.\n\n"
        "The customer is in a conversation with a seller about a men's watch. The seller is selling it for $30.\n"
        "The customer wants to clarify about the material of the strap. The customer asks\n"
        "Could you please let me know the material of the strap?\n\n")

    prompt_ = prompt_ + "A customer is in a conversation with a seller about a " + \
        product_feature + ". The seller is selling it for $" + seller_price + ".\n"
    prompt_ = prompt_ + "The customer wants to clarify " + \
        clarification_feature + ". The customer asks\n"

    return gen_utterance(prompt_, temperature, max_len)


def resolve_clarification(product_feature,
                          customer_utterance,
                          temperature=0.6,
                          max_len=850):

    prompt_ = (
        "You are a seller and you know the following details about a product. The product is a laptop called Dell X8 it has 16 GB ram, 11-inch screen and Intel i7 processor.\n"
        '''A customer asks, "Is the RAM of the Dell X8 laptop 16 GB as advertised?"\n'''
        "You reply.\n"
        "<start> Yes, the Dell X8 laptop has 16 GB of RAM as advertised. I'm also happy to help you with any other queries that you may have.\n\n"
        "You are a seller and you know the following details about a product. The product is a bedsheet, made with cotton, has blue and yellow designs and the material is extremely breathable.\n"
        '''A customer asks, "How is the material like?"\n'''
        "You reply.\n"
        "<start> The bedsheet is made out of cotton and is highly breathable making it very comfortable.\n\n"
        "You are a seller and you know the following details about a product. The product is a men's watch with a leather strap.\n"
        '''A customer asks, "Could you please let me know the material of the strap?"\n'''
        "You reply.\n"
        "<start> The material of the strap is leather. Is there any other query that I can help you with?\n\n"
        "You are a seller and you know the following details about a product. The product is a headphone called Beats Solo 3, it has noise cancellation, is over the ear, has soft ear cushions and a good audio quality.\n"
        '''A customer asks, "Can you tell me more about the audio quality of the Beats Solo 3 headphones? I'm interested in purchasing them, but I want to make sure they have good sound."\n'''
        "You reply.\n"
        "<start> These headphones have a great audio quality with noise cancellation, over the ear design and soft ear cushions to provide a comfortable experience. You won't be disappointed with the sound quality they offer.\n\n"
        "You are a seller and you know the following details about a product. The product is a black school bag, the brand is bumblebee, it has 3 sections and can carry many books.\n"
        '''A customer asks, "Which brand?"\n'''
        "You reply.\n"
        "<start> The brand of the schoolbag is Bumblebee. Can I help you with anything else?\n\n"
    )

    prompt_ = prompt_ + "You are a seller and you know the following details about a product. The product is a " + \
        product_feature + ".\n"
    prompt_ = prompt_ + f'''A customer asks, "{customer_utterance}"\n'''
    prompt_ = prompt_ + "You reply.\n"

    return gen_utterance(prompt_, temperature, max_len)


def accept_seller(product_feature,
                  customer_initial,
                  customer_utterance,
                  temperature=0.7,
                  max_len=1000):

    prompt_ = (
        "A seller is negotiating with a customer for a laptop called Dell X8, it has 16 GB ram, 11-inch screen and Intel i7 processor.\n"
        '''The customer tells "I appreciate that you need to make a profit on this item, but unfortunately, $770 is above my budget for a laptop. I'm willing to negotiate up to $730 if necessary. Is there any way you could lower the price to meet me somewhere in the middle?"\n'''
        "The seller agrees to the customer's price of $730 by saying,\n"
        "<start> Thank you for your offer, I understand your budget concerns. I am willing to accept your offer of $730 for the Dell X8 laptop. Let's make the deal happen.\n\n"
        "A seller is negotiating with a customer for a headphone called Beats Solo 3, it has noise cancellation, is over the ear, and has soft ear cushions.\n"
        '''The customer tells "My budget is $120. Can you give a discount and sell it for $120? Let me know if that's possible."\n'''
        "The seller agrees to the customer's price of $120 by saying,\n"
        "<start> I can accept your offer of $120 for the Beats Solo 3 headphones. I believe it's a fair price for both of us. Let's proceed with the purchase.\n\n"
        "A seller is negotiating with a customer for a men's watch.\n"
        '''The customer tells, "Sorry, $25 is above my budget. Can you meet me in the middle and sell for $22?"\n'''
        "The seller agrees to the customer's price of $25 by saying,\n"
        "<start> Sure, I can meet you halfway and sell the watch for $22. Let's make a deal.\n\n"
    )

    prompt_ = prompt_ + "A seller is negotiating with a customer for a " + \
        product_feature + ".\n"
    prompt_ = prompt_ + f'''The customer tells "{customer_utterance}"\n''' + \
        f"The seller agrees to the customer's price of {customer_initial} by saying,\n"

    return gen_utterance(prompt_, temperature, max_len)


def accept_customer(product_feature,
                    seller_initial,
                    seller_utterance,
                    temperature=0.7,
                    max_len=650):

    prompt_ = (
        "A customer is negotiating with a seller for a laptop called Dell X8, it has 16 GB ram, 11-inch screen and Intel i7 processor.\n"
        '''The seller tells "While I appreciate your offer, I cannot go as low as $700. However, I am willing to sell it to you for $720, which is the best price I can offer. This laptop has high-end features such as a 16 GB RAM, 11-inch screen and an Intel i7 processor, and I'm confident it will meet your needs. Please let me know if you would like to purchase it at this price."\n'''
        "The customer agrees to the seller's price of $720 and says,\n"
        "<start> You make a good point about the laptop's features, and I agree that $720 is a fair price. I'm willing to purchase it at that price. Let's proceed with the transaction.\n\n"
        "A customer is negotiating with a seller for a headphone called Beats Solo 3, it has noise cancellation, is over the ear, and has soft ear cushions.\n"
        '''The seller tells "While our ideal price for the Beats Solo 3 headphones is $130, I understand your budget and am willing to meet you in the middle at $120. Unfortunately, I cannot go lower than that. I hope we can still come to a mutually beneficial agreement."\n'''
        "The customer agrees to the seller's price of $120 and says,\n"
        "<start> Okay, thank you. I'll buy the headphones for $120.\n\n"
        "A customer is negotiating with a seller for a men's watch.\n"
        '''The seller tells "I understand that $20 is your budget, but the best price I can offer for this watch is $22. It is a high-quality product that is worth the investment. Shall we shake on this?"\n'''
        "The customer agrees to the seller's price of $22 and says,\n"
        "<start> I agree, let's go ahead with the purchase for $22.\n\n")

    prompt_ = prompt_ + "A customer is negotiating with a seller for a " + \
        product_feature + ".\n"
    prompt_ = prompt_ + f'''The seller tells "{seller_utterance}"\n''' + \
        f"The customer agrees to the seller's price of ${seller_initial} and says,\n"

    return gen_utterance(prompt_, temperature, max_len)


def ack_accept_customer(seller_utterance, temperature=0.7, max_len=1000):

    prompt_ = (
        "A seller has agreed to the offer set by a customer, the customer wants to respond to the seller.\n"
        '''The seller says, "Thank you for your offer, I understand your budget concerns. I am willing to accept your offer of $730 for the Dell X8 laptop. Let's make the deal happen."\n'''
        "The customer agrees to the seller by saying.\n"
        "<start> Thank you for understanding my budget requirements, we can proceed with the deal!\n\n"
        "A seller has agreed to the offer set by a customer, the customer wants to respond to the seller.\n"
        '''The seller says, "I can accept your offer of $120 for the Beats Solo 3 headphones. I believe it's a fair price for both of us. Let's proceed with the purchase."\n'''
        "The customer agrees to the seller by saying.\n"
        "<start> Yes, we can proceed with the purchase.\n\n"
        "A seller has agreed to the offer set by a customer, the customer wants to respond to the seller.\n"
        '''The seller says, "Thank you for your offer of $450 for the Samsung Galaxy S21. I agree that it is a fair price, and we can proceed with the transaction."\n'''
        "The customer agrees to the seller by saying.\n"
        "<start> Great, thank you for accepting my offer. Let's proceed with the transaction!\n\n"
        "A seller has agreed to the offer set by a customer, the customer wants to respond to the seller.\n"
        '''The seller says, "I understand that you were looking for a discount, and I am willing to accept your offer of $300 for the Canon EOS camera. Let's proceed with the transaction."\n'''
        "The customer agrees to the seller by saying.\n"
        "<start> Thank you for accepting my offer, I'm excited to purchase the camera. Let's proceed with the transaction!\n\n"
        "A seller has agreed to the offer set by a customer, the customer wants to respond to the seller.\n"
    )

    prompt_ = prompt_ + \
        f'''The seller says, {seller_utterance}\n''' + \
        "The customer agrees to the seller by saying.\n"

    return gen_utterance(prompt_, temperature, max_len)


def ack_accept_seller(customer_utterance, temperature=0.7, max_len=1000):

    prompt_ = (
        "A customer has agreed to purchase a product from a seller, the seller wants to thank the customer and proceed with the transaction.\n"
        '''The customer says, "You make a good point about the laptop's features, and I agree that $720 is a fair price. I'm willing to purchase it at that price. Let's proceed with the transaction"\n'''
        "The seller thanks the customer and proceeds with the deal by saying.\n"
        "<start> Thank you for accepting the offer, I'm sure that you will be happy with the laptop. We can now proceed with the transaction!\n\n"
        "A customer has agreed to purchase a product from a seller, the seller wants to thank the customer and proceed with the transaction.\n"
        '''The customer says, "Okay, thank you. I'll buy them for $120."\n'''
        "The seller thanks the customer and proceeds with the deal by saying.\n"
        "<start> That's great, $120 will be a fruitful deal for both of us, let us go ahead with the transaction.\n\n"
        "A customer has agreed to purchase a product from a seller, the seller wants to thank the customer and proceed with the transaction.\n"
        '''The customer says, "I have been looking for a product with these specifications and I think your offer is reasonable. I'm willing to pay $500 for it. Let's proceed with the transaction."\n'''
        "The seller thanks the customer and proceeds with the deal by saying.\n"
        "<start> Thank you for considering my offer, I am confident that the product will meet your expectations. We can now proceed with the transaction!\n\n"
        "A customer has agreed to purchase a product from a seller, the seller wants to thank the customer and proceed with the transaction.\n"
        '''The customer says, "I am happy with the quality of the product and I think the price is fair. I'm willing to buy it for $900. Let's proceed with the transaction."\n'''
        "The seller thanks the customer and proceeds with the deal by saying.\n"
        "<start> I appreciate your interest in the product, and I am glad that you find the price reasonable. Let's proceed with the transaction, and I am sure you will be satisfied with your purchase!\n\n"
        "A customer has agreed to purchase a product from a seller, the seller wants to thank the customer and proceed with the transaction.\n"
    )

    prompt_ = prompt_ + \
        f'''The customer says, {customer_utterance}\n''' + \
        "The seller replies by saying.\n"

    return gen_utterance(prompt_, temperature, max_len)


def reject_seller(product_feature,
                  seller_intial,
                  seller_final,
                  customer_utterance,
                  temperature=0.7,
                  max_len=1000):
    prompt_ = (
        "A seller is negotiating with a customer for a laptop called Dell X8, it has 16 GB ram, 11-inch screen and Intel i7 processor. The seller is willing to go down to $720.\n"
        '''The customer tells "I understand that you need to make a profit on this item, but unfortunately it is above my budget. I was ideally hoping to purchase the Dell X8 for $600, but I'm willing to negotiate up to $650 if necessary. Is there any way you could lower the price to meet me somewhere in the middle?"\n'''
        "The seller rejects the offer by saying.\n"
        "<start> While I appreciate your offer, I cannot go as low as $650. This laptop has high-end features such as a 16 GB RAM, 11-inch screen and an Intel i7 processor, and I'm confident it will meet your needs. Please let me know if you would like to purchase it at this price. Thank you for your interest.\n\n"
        "A seller is negotiating with a customer for a headphone called Beats Solo 3, it has noise cancellation, is over the ear, and has soft ear cushions. The seller is willing to go down to $120.\n"
        '''The customer tells "My budget is $90. Can you give a discount and sell it for $90? Let me know if that's possible."\n'''
        "The seller rejects the offer by saying.\n"
        "<start> Unfortunately, we cannot sell the Beats Solo 3 for $90. The best I can do is $120 which is already a great deal considering the features of the headphone. If you are willing to reconsider your offer, we would be happy to work with you to find a price that suits your budget. Thank you for your interest.\n\n"
        "A seller is negotiating with a customer for a men's watch. The seller is willing to go down to $20.\n"
        '''The customer tells, "Sorry, $25 is above my budget. Can you sell it for $10?"\n'''
        "The seller rejects the offer by saying.\n"
        "<start> I'm sorry, I will not be able to sell you the watch for $10, the lowest I can go is $20. Thank you for your offer.\n\n"
    )

    prompt_ = prompt_ + "A seller is negotiating with a customer for a " + product_feature + \
        ".The seller is willing to go down to $" + seller_final + ".\n"
    prompt_ = prompt_ + \
        f'''The customer tells "{customer_utterance}"\n''' + \
        "The seller rejects the offer by saying.\n"

    return gen_utterance(prompt_, temperature, max_len)


def reject_customer(product_feature,
                    seller_initial,
                    customer_final,
                    seller_utterance,
                    temperature=0.7,
                    max_len=1000):

    prompt_ = (
        "A customer is negotiating with a seller for a laptop called Dell X8, it has 16 GB ram, 11-inch screen and Intel i7 processor.\n"
        '''The seller tells "While I appreciate your offer, I cannot go as low as $700, $750 is the best price that I can offer. This laptop has high-end features such as a 16 GB RAM, 11-inch screen and an Intel i7 processor, and I'm confident it will meet your needs. Please let me know if you would like to purchase it at this price. Thank you for your interest"\n'''
        "The customer's budget is $700 and cannot go up to $750. The customer rejects the offer by saying.\n"
        "<start> I cannot accept your offer of $750, since I'm limited to my budget of $700. Let me know if you change your mind or have any other deals. Thank you.\n\n"
        "A customer is negotiating with a seller for a headphone called Beats Solo 3, it has noise cancellation, is over the ear, and has soft ear cushions.\n"
        '''The seller tells "While our ideal price for the Beats Solo 3 headphones is $150, I understand your budget and am willing to go down to $140. Unfortunately, I cannot go lower than that. I hope we can still come to a mutually beneficial agreement."\n'''
        "The customer's budget is $120 and cannot go up to $140. The customer rejects the offer by saying.\n"
        "<start> Thank you for considering my offer. However, my budget is limited to $120, and I cannot go higher than that to $140. Good bye\n\n"
        "A customer is negotiating with a seller for a men's watch.\n"
        '''The seller tells "I understand that $20 is your budget, but the best price I can offer for this watch is $30. It is a high-quality product that is worth the investment. Shall we shake on this?"\n'''
        "The customer's budget is $22 and cannot go up to $30. The customer rejects the offer by saying.\n"
        "<start> My budget is limited to $22, and I cannot go beyond that. I'll look for other options. Thank you.\n\n"
    )

    prompt_ = prompt_ + "A customer is negotiating with a seller for a " + \
        product_feature + ".\n"
    prompt_ = prompt_ + f'''The seller tells "{seller_utterance}"\n''' + \
        f"The customer's budget is ${customer_final} and cannot go up to ${seller_initial}. The customer rejects the offer by saying.\n"

    return gen_utterance(prompt_, temperature, max_len)


def avoid_rejection(customer_utterance, temperature=0.7, max_len=950):

    # Change this to include the buyer's last price rather than thinking of customer's utterance
    prompt_ = (
        "A seller is negotiating with a customer on a product. The customer has hit his budget and is not willing to negotiate further, the seller is willing to lower his price to meet the customer's budget.\n"
        '''The customer says, "My budget is limited to $22, and I cannot go beyond that. I'll look for other options. Thank you."\n'''
        "The seller agrees to the customer's deal by saying.\n"
        "<start> Ok, in that case we can proceed with $22. We understand your budget limitation. Shall we go ahead with the purchase?\n\n"
        "A seller is negotiating with a customer on a product. The customer has hit his budget and is not willing to negotiate further, the seller is willing to lower his price to meet the customer's budget.\n"
        '''The customer says, "Thank you for considering my offer. However, I cannot go higher than $120."\n'''
        "The seller agrees to the customer's deal by saying.\n"
        "<start> Alright, we can go forward with $120 then. Shall we proceed to checkout? \n\n"
        "A seller is negotiating with a customer on a product. The customer has hit his budget and is not willing to negotiate further, the seller is willing to lower his price to meet the customer's budget.\n"
        '''The customer says, "I'm interested in purchasing this product, but I cannot go more than $40."\n'''
        "The seller agrees to the customer's request by saying.\n"
        "<start> Okay, in that case I'm willing to meet you at $40, shall we proceed with the payment?\n\n"
        "A seller is negotiating with a customer on a product. The customer has hit his budget and is not willing to negotiate further, the seller is willing to lower his price to meet the customer's budget.\n"
        '''The customer says, "I will have to look for other products since $50 is the highest that i can go."\n'''
        "The seller agrees to the customer's request by saying.\n"
        "<start> Alright, we understand your budget limitations and we're willing to meet you at $50. Would you like to proceed with the purchase?\n\n"
        "A seller is negotiating with a customer on a product. The customer has hit his budget and is not willing to negotiate further, the seller is willing to lower his price to meet the customer's budget.\n"
    )

    prompt_ = prompt_ + f'''The customer says, "{customer_utterance}"\n''' + \
        "The seller agrees to the customer's request by saying.\n"

    return gen_utterance(prompt_, temperature, max_len)


def avoid_rejection_v2(customer_initial, temperature=0.7, max_len=650):
    prompt_ = (
        "A customer says that he cannot afford more than 20 on the product that a seller is selling.\n"
        "Seeing this, the seller says that he will sell the product for 20 by saying, \n"
        "<start> Okay, I understand your budget limitations and I will meet you at your price of 20, shall we proceed with the deal?\n\n"
        "A customer says that he cannot afford more than $135 on the product that a seller is selling.\n"
        "Seeing this, the seller says that he will sell the product for $135 by saying, \n"
        "<start> Alright, in that case I can sell you the product for $135!\n\n"
        "A customer says that he cannot afford more than $560 on the product that a seller is selling.\n"
        "Seeing this, the seller says that he will sell the product for $560 by saying, \n"
        "<start> Okay I'll sell you the product for $560, shall we go ahead with this?\n\n"
        "A customer says that he cannot afford more than $36 on the product that a seller is selling.\n"
        "Seeing this, the seller says that he will sell the product for $36 by saying, \n"
        "<start> Alright, 36 is fine with me, shall we proceed with the transaction?\n\n"
    )

    prompt_ = prompt_ + f"A customer says that he cannot afford more than ${customer_initial} on the product that a seller is selling. \n" + \
        f"Seeing this, the seller says that he will sell the product for ${customer_initial} by saying,\n"

    return gen_utterance(prompt_, temperature, max_len)


def negotiate_add(
    product_features,
    addition,
    temperature=0.7,
    max_len=1050,
):

    prompt_ = (
        "A customer wants to ask a seller about the price of a product if an add-on is added back again to the product.\n"
        "The product is a laptop called Dell X8 it has 16 GB ram, 11-inch screen and Intel i7 processor.\n"
        "The add-on to be added is a gaming mouse, the customer says.\n"
        "<start> What will be the price of the bundle if I buy the laptop along with the gaming mouse? \n\n"
        "A customer wants to ask a seller about the price of a product if an add-on is added back again to the product.\n"
        "The product is a headphone called Beats Solo 3, it has noise cancellation, is over the ear, and has soft ear cushions.\n"
        "The add-on to be added is a keyboard.\n"
        "<start> What is your offer if I also buy the keyboard? \n\n"
        "A customer wants to ask a seller about the price of a product if an add-on is added back again to the product.\n"
        "The product is a television called Samsung QN90A, it has a 65-inch screen, 4K resolution, and built-in Alexa.\n"
        "The add-on to be added is a soundbar, the customer says.\n"
        "<start> What will be the total price if I buy the TV along with the soundbar?\n\n"
        "A customer wants to ask a seller about the price of a product if an add-on is added back again to the product.\n"
        "The product is a Borwn coffee mug.\n"
        "The add-on to be added is a Saucer, the customer says.\n"
        "<start> What is your best deal if I were to buy the Saucer as well?\n\n"
        "A customer wants to ask a seller about the price of a product if an add-on is added back again to the product.\n"
    )

    prompt_ = prompt_ + f"The product is a {product_features}\n" + \
        f"The add-on to be added is a {addition}, the customer says.\n"

    return gen_utterance(prompt_, temperature, max_len)


def negotiate_remove(product_features,
                     removal,
                     seller_price,
                     temperature=0.7,
                     max_len=900,
                     additional_context=""):

    prompt_ = (
        "A customer is negotiating with a seller about a product. The customer wants to ask for another deal to the seller.\n"
        "The initial deal was a laptop called Dell X8, it has 16 GB ram, 11-inch screen and Intel i7 processor along with a gaming mouse. The price for this deal was $800. The customer wants to remove the gaming mouse from the deal.\n"
        "The customer asks for the new deal by saying.\n"
        "<start> I do not really need the mouse, is it possible to just sell me the laptop?\n\n"
        "A customer is negotiating with a seller about a product. The customer wants to ask for another deal to the seller.\n"
        "The initial deal was a mens watch, with delivery and warranty. The price for this deal was $100. The customer wants to remove the delivery from the deal.\n"
        "The customer asks for the new deal by saying.\n"
        "<start> Hey, I do not want the watch to be delivered to me, I will pick it up, what can you offer for this?\n\n"
        "A customer is negotiating with a seller about a product. The customer wants to ask for another deal to the seller.\n"
        "The initial deal was a smart TV, with 4K resolution, 50-inch screen and built-in speakers. The price for this deal was $1000, including a wall mount. The customer wants to remove the wall mount from the deal.\n"
        "The customer asks for the new deal by saying.\n"
        "<start> Excuse me, I don't need the wall mount included in the package. Can we modify the deal and remove the wall mount? How much would it cost? \n\n"
        "A customer is negotiating with a seller about a product. The customer wants to ask for another deal to the seller.\n"
        "The initial deal was a set of golf clubs that included a driver, 3-wood, 5-wood, irons 4-PW, and a putter. The price for this deal was $500. The customer wants to remove the putter from the deal.\n"
        "The customer asks for the new deal by saying.\n"
        "<start> Hello, I am interested in buying the set of golf clubs but I do not need the putter. Is it possible to buy the set without the putter? What would be the new price?\n\n"
    )

    prompt_ = prompt_ + "A customer is negotiating with a seller about a product. The customer wants to ask for another deal to the seller. " + additional_context + "\n"
    prompt_ = prompt_ + "The initial deal was a " + product_features + ". The price for this deal was $" + \
        seller_price + ". The customer wants to remove the " + \
        removal + " from the deal.\n "
    prompt_ = prompt_ + "The customer asks for the new deal by saying.\n"

    return gen_utterance(prompt_, temperature, max_len)


def negotiate_remove_delivery(product_features,
                              seller_price,
                              temperature=0.7,
                              max_len=900,
                              additional_context=""):
    return negotiate_remove(product_features,
                            "delivery",
                            seller_price,
                            max_len=max_len,
                            temperature=temperature,
                            additional_context=additional_context)


def negotiate_remove_decrease_price(product_features,
                                    addition,
                                    customer_price,
                                    seller_price,
                                    temperature=0.7,
                                    max_len=900,
                                    additional_context=""):

    prompt_ = (
        "A customer is negotiating with a seller about a product. The customer wants to ask for another deal to the seller.\n"
        "The initial deal was a laptop called Dell X8, it has 16 GB ram, 11-inch screen and Intel i7 processor along with a gaming mouse. The customer wants to remove the gaming mouse from the deal.\n"
        "The customer asks for the new deal and pitches a price of $750 by saying.\n"
        "<start> I do not really need the mouse, is it possible to just sell me the laptop for $750?\n\n"
        "A customer is negotiating with a seller about a product. The customer wants to ask for another deal to the seller.\n"
        "The initial deal was a mens watch, with delivery and warranty. The customer wants to remove the delivery from the deal.\n"
        "The customer asks for the new deal and pitches a price of $70 by saying.\n"
        "<start> Hey, I do not want the watch to be delivered to me, I will pick it up. Can you sell it to me for $70?\n\n"
        "A customer is negotiating with a seller about a product. The customer wants to ask for another deal to the seller.\n"
        "The initial deal was a sofa set, consisting of a 3-seater, a 2-seater, and an armchair. The customer wants to remove the armchair from the deal.\n"
        "The customer asks for the new deal and pitches a price of $1700 by saying.\n"
        "<start> I really like the sofa set but I don't have enough space for the armchair. Can you sell me the 3-seater and 2-seater for $1700?\n\n"
        "A customer is negotiating with a seller about a product. The customer wants to ask for another deal to the seller.\n"
        "The initial deal was a gym membership for 6 months, which included personal training sessions. The customer wants to remove the personal training sessions from the deal.\n"
        "The customer asks for the new deal and pitches a price of $800 by saying.\n"
        "<start> I am on a tight budget and I cannot afford personal training sessions. Can you sell me the gym membership for 6 months without the personal training sessions for $800?\n\n"
    )

    prompt_ = prompt_ + "A customer is negotiating with a seller about a product. The customer wants to ask for another deal to the seller. " + additional_context + "\n"
    prompt_ = prompt_ + "The initial deal was a " + product_features + \
        ". The customer wants to remove the " + addition + " from the deal.\n "
    prompt_ = prompt_ + \
        f"The customer asks for the new deal and pitches a price of ${customer_price} by saying.\n"

    return gen_utterance(prompt_, temperature, max_len)


def negotiate_price_remove(product_features,
                           customer_utterance,
                           seller_final,
                           temperature=0.7,
                           max_len=1500):
    # Reply to negotiate-remove, negotiate-remove_decrease_price

    prompt_ = (
        "A customer asks a seller about removing an item from the seller's product bundle. Produce the seller's reply according to the given information.\n"
        "The product the seller is selling is a laptop called Dell X8, it has 16 GB ram, 11-inch screen and Intel i7 processor along with a gaming mouse.\n"
        '''The customer's query was, "I do not really need the mouse, is it possible to just sell me the laptop?"\n'''
        "The seller is willing to proceed with the deal for $500, the seller replies.\n"
        "<start> Sure, I can sell you just the mouse for $500.\n\n"
        "A customer asks a seller about removing an item from the seller's product bundle. Produce the seller's reply according to the given information.\n"
        "The product the seller is selling is sofa set, consisting of a 3-seater, a 2-seater, and an armchair.\n"
        '''The customer's query was, "I really like the sofa set but I don't have enough space for the armchair. Can you sell me the 3-seater and 2-seater for $1700?"\n'''
        "The seller is willing to proceed with the deal for $2000, the seller replies.\n"
        "<start> Oh I see, I can sell you just the 3-seater and the 2-seater however, $1700 is too less. How about $2000?\n\n"
        "A customer asks a seller about removing an item from the seller's product bundle. Produce the seller's reply according to the given information.\n"
        "The product the seller is selling is a camera bundle, which includes a camera, two lenses, a tripod, and a camera bag.\n"
        '''The customer's query was, "I already have a tripod, can I purchase the camera bundle without it?"\n'''
        "The seller is willing to proceed with the deal for $1000, the seller replies.\n"
        "<start> Certainly! I can remove the tripod from the bundle and sell you the camera, two lenses, and camera bag for $1000.\n\n"
        "A customer asks a seller about removing an item from the seller's product bundle. Produce the seller's reply according to the given information.\n"
        "The product the seller is selling is a home theater system, which includes a soundbar, a subwoofer, and two satellite speakers.\n"
        '''The customer's query was, "I'm interested in buying the home theater system, but I don't need the subwoofer. Can I purchase just the soundbar and two satellite speakers for 400?"\n'''
        "The seller is willing to proceed with the deal for $600, the seller replies.\n"
        "<start> Absolutely! I can remove the subwoofer from the bundle and sell you the soundbar and two satellite speakers, however $400 is too less, how about $600?\n\n"
        "A customer asks a seller about removing an item from the seller's product bundle. Produce the seller's reply according to the given information.\n"
    )

    prompt_ = prompt_ + f"The product the seller is selling is a {product_features}\n" + f'''The customer's query was, "{customer_utterance}"\n.''' + \
        f"The seller is willing to proceed with the deal for ${seller_final}, the seller replies\n"

    return gen_utterance(prompt_, temperature, max_len)


def negotiate_price_remove_v2(product_features,
                              customer_utterance,
                              seller_final,
                              temperature=0.7,
                              max_len=1500):

    prompt_ = (
        "A customer asks a seller a query, produce the reply.\n"
        '''The customer's query was, "I do not really need the mouse, is it possible to just sell me the laptop for $300?"\n'''
        "The seller is willing to proceed with the deal for $500, the seller replies.\n"
        "<start> Sure, I can sell you just the laptop, but it'll cost $500. I'm sure that the laptop will be well worth the investment.\n\n"
        "A customer asks a seller a query, produce the reply.\n"
        '''The customer's query was, "I really like the sofa set but I don't have enough space for the armchair. Can you sell me the 3-seater and 2-seater for $1700?"\n'''
        "The seller is willing to proceed with the deal for $2000, the seller replies.\n"
        "<start> Oh I see, I can sell you just the 3-seater and the 2-seater however, $1700 is too less. How about $2000?\n\n"
        "A customer asks a seller a query, produce the reply.\n"
        '''The customer's query was, "I already have a tripod, can I purchase the camera bundle without it for lets say $700?"\n'''
        "The seller is willing to proceed with the deal for $1000, the seller replies.\n"
        "<start> Certainly! I can remove the tripod from the bundle and sell you the camera, two lenses, and camera bag but it'll be for $1000. I'm sorry I couldn't match your offer of $700, I'm confident that the product is worth $1000 and I'm sure you would be happy with the product!\n\n"
        "A customer asks a seller a query, produce the reply.\n"
        '''The customer's query was, "I'm interested in buying the home theater system, but I don't need the subwoofer. Can I purchase just the soundbar and two satellite speakers for $400?"\n'''
        "The seller is willing to proceed with the deal for $600, the seller replies.\n"
        "<start> Absolutely! I can remove the subwoofer from the bundle and sell you the soundbar and two satellite speakers, however $400 is too less, how about $600?\n\n"
        "A customer asks a seller a query, produce the reply.\n")

    prompt_ = prompt_ + f'''The customer's query was, "{customer_utterance}"\n.''' + \
        f"The seller is willing to proceed with the deal for ${seller_final}, the seller replies\n"

    return gen_utterance(prompt_, temperature, max_len)


def negotiate_noChange_seller(product_feature,
                              seller_initial,
                              customer_utterance,
                              temperature=0.5,
                              max_len=1200):
    # Needs to be there from both buyer and seller
    # Currently it is for seller

    prompt_ = (
        "A seller is negotiating with a customer for a laptop called Dell X8, it has 16 GB ram, 11-inch screen and Intel i7 processor. The seller ideally wants it for $770 and is not willing to reduce the price.\n"
        '''The customer tells "I appreciate that you need to make a profit on this item, but unfortunately, $770 is above my budget for a laptop. I was ideally hoping to purchase the Dell X8 for $500, but I'm willing to negotiate up to $570 if necessary. Is there any way you could lower the price to meet me somewhere in the middle?"\n'''
        "(Remember, the seller cannot go lower than $770) The seller endorses the product by saying.\n"
        "<start> While I appreciate your offer, I cannot go as low as $570. I cannot lower the price further since the laptop is high-end and is well worth $770. It has 16 GB ram and an Intel i7 processor, making it ideal for heavy duty applications. I'm sure that you would be pleased with it even for $770!\n\n"
        "A seller is negotiating with a customer for a headphone called Beats Solo 3, it has noise cancellation, is over the ear, and has soft ear cushions. The seller ideally wants it for $130 and is not willing to reduce the price.\n"
        '''The customer tells "My budget is $110. Can you give a discount and sell it for $110? Let me know if that's possible."\n'''
        "(Remember, the seller cannot go lower than $130) The seller replies by saying.\n"
        "<start> I understand your budget, however, I'm sorry I cannot reduce the price of the headphones since it has great features such as noise cancellation and is also very comfortable to wear due to its over the ear design and soft ear cushions. I'm sure that it is worth $130.\n\n"
        "A seller is negotiating with a customer for a table lamp called BrightLite, it has a touch control feature, adjustable brightness, and a sleek design. The seller ideally wants it for $70 and is not willing to reduce the price.\n"
        '''The customer tells "I really like the BrightLite table lamp, but $70 is above my budget. I was ideally hoping to purchase it for $50, but I'm willing to negotiate up to $60. Is there any way you could lower the price to meet me somewhere in the middle?"\n'''
        "(Remember, the seller cannot go lower than $70) The seller responds by saying.\n"
        "<start> While I appreciate your offer, I cannot go as low as $60. The BrightLite table lamp is a high-quality product and is well worth the price of $70. It has a touch control feature, adjustable brightness, and a sleek design, making it an ideal choice for modern home decor. I'm sure that you would be pleased with it even for $70!\n\n"
        "A seller is negotiating with a customer for a smartwatch called Apple Watch Series 7, it has a retina display, ECG monitoring, and fall detection. The seller ideally wants it for $500 and is not willing to reduce the price.\n"
        '''The customer tells "I am interested in buying the Apple Watch Series 7, but I am on a tight budget. I can only afford to pay $400 for it. Can you please give me a discount?"\n'''
        "(Remember, the seller cannot go lower than $500) The seller replies by saying.\n"
        "<start> I understand your budget constraints, but I'm sorry I cannot reduce the price of the Apple Watch Series 7. It has a retina display, ECG monitoring, and fall detection features that make it worth $500. It is also one of the latest models from Apple, which adds to its value. I'm sure that it will be worth the investment even for $500!\n\n"
    )

    inform_feature = "A seller is negotiating with a customer for a " + product_feature + \
        ". The seller ideally wants it for $" + seller_initial + \
        " and is not willing to reduce the price.\n"
    customer_utt = f'''The customer tells "{customer_utterance}"\n'''

    prompt_ = prompt_ + inform_feature + customer_utt + \
        f"(Remember, the seller cannot go lower than ${seller_initial}) " + \
        "The seller replies by saying.\n"

    return gen_utterance(prompt_, temperature, max_len)


def negotiate_noChange_buyer(seller_price,
                             customer_price,
                             temperature=0.7,
                             max_len=700):
    prompt_ = (
        "A seller asks a customer to buy a product for $500.\n"
        "The customer can go up to $430.\n"
        "The customer replies to the seller, saying that he cannot increase his budget further by saying,\n"
        "<start> I appreciate the offer of $500, but I cannot go more than $430.\n\n"
        "A seller asks a customer to buy a product for $340.\n"
        "The customer can go up to $200.\n"
        "The customer replies to the seller, saying that he cannot increase his budget further by saying,\n"
        "<start> $200 is the maximum that I can go, I will not be able to afford $340.\n\n"
        "A seller asks a customer to buy a product for $344.\n"
        "The customer can go up to $238.\n"
        "The customer replies to the seller, saying that he cannot increase his budget further by saying,\n"
        "<start> I can only go up to $238, I cannot increase it further, sorry.\n\n"
        "A seller asks a customer to buy a product for $23.\n"
        "The customer can go up to $7.\n"
        "The customer replies to the seller, saying that he cannot increase his budget further by saying,\n"
        "<start> $23 seems out of my budget for this product, I can only go up to $7.\n\n"
    )
    prompt_ = prompt_ + f"A seller asks a customer to buy a product for ${seller_price}.\n" + f"The customer can go up to ${customer_price}.\n" + \
        "The customer replies to the seller, saying that he cannot increase his budget further by saying,\n"

    return gen_utterance(prompt_, temperature, max_len)


def greet_inform_noChange(product_feature,
                          seller_initial,
                          customers_utterance,
                          temperature=0.7,
                          max_len=1100):

    prompt_ = (
        "A seller is selling a laptop called Dell X8, it has 16 GB ram, 11-inch screen and Intel i7 processor. The product is being sold online.\n"
        '''The customer asks, "Hi, I'm interested in buying the laptop that you have listed for my son. I was hoping to get it for $500."\n'''
        "The seller is selling it for $800 and is not willing to decrease the price, the seller informs this to the customer by saying: \n"
        "<start> Hello! I'm sure your son would love the laptop, but I'm sorry the lowest I cannot reduce the price. I'm sure you can understand that I need to make a profit on this item, so $500 is not an option, furthermore the laptop is high end and is worth $800. Let me know if you'd like to purchase it at $800.\n\n"
        "A seller is selling a headphone called Beats Solo 3, it has noise cancellation, is over the ear, and has soft ear cushions. The product is being sold online.\n"
        '''The customer asks, "Hello, I am interested in buying your Beats Solo 3 headphones, but I was hoping to get a better price than $149. Is there anything you can do to lower the price to $100?"\n'''
        "The seller seller is selling it for $149 and is not willing to decrease the price, the seller informs this to the customer by saying: \n"
        "<start> Hello there! Thank you for your interest in the Beats Solo 3 headphones. While I appreciate your offer, I'm sorry I cannot lower the price to $100. The headphones have great features such as noise cancellation, over the ear design, and soft ear cushions that make it worth the price of $149. This is the best price that I can offer at this time.\n\n"
        "A seller is selling a dress, it is a red knee-length dress made of silk material with a beautiful floral pattern. The product is being sold online.\n"
        '''The customer asks, "Hi, I'm interested in buying the red silk dress that you have listed. I was hoping to get it for $50."\n'''
        "The seller is selling it for $80 and is not willing to decrease the price, the seller informs this to the customer by saying: \n"
        "<start> Hello! I'm glad to hear that you're interested in the red silk dress. I'm sorry I cannot go as low as $50. This dress is made of high-quality silk material and has a beautiful floral pattern that makes it a great addition to any wardrobe. Would you be willing to purchase it for $80? Let me know if this works for you.\n\n"
        "A seller is selling a bicycle called Giant XTC Advanced 29, it has a carbon frame, hydraulic disc brakes, and a 12-speed drivetrain. The product is being sold online.\n"
        '''The customer asks, "Hi, I'm interested in buying the Giant XTC Advanced 29 bike. I was hoping to get it for $2000."\n'''
        "The seller is selling it for $3000 and is not willing to decrease the price, the seller informs this to the customer by saying: \n"
        "<start> Hello! Thank you for your interest in the Giant XTC Advanced 29 bike. While I understand your budget, I'm sorry I cannot sell it for $2000. This bike is made of high-quality materials, has advanced features like hydraulic disc brakes and a 12-speed drivetrain that make it worth the price of $3000. Let me know if you'd like to purchase it at $3000.\n\n"
    )

    inform_feature = "A seller is selling a " + \
        product_feature + ". The product is being sold online.\n"
    prompt_ = prompt_ + inform_feature + f'''The customer asks,"{customers_utterance}"\n''' + "The seller is selling it for $" + \
        seller_initial + " and is not willing to decrease the price, the seller informs this to the customer by saying: \n"

    return gen_utterance(prompt_, temperature, max_len)


def tell_price(sellers_final,
               customers_utterance,
               temperature=0.7,
               max_len=1000):

    # This should be the reply for both remove X and Add X
    prompt_ = (
        "A customer is asking a seller the price of a bundle of products, what would be the seller's reply?\n"
        '''The customer says, "What would be the price if I wanted just the laptop and not the charger?" \n'''
        "The seller would sell the bundle for $1000, the seller replies by saying.\n"
        "<start> Well if you want just the laptop and not the charger it would cost you $1000, how does that sound? \n\n"
        "A customer is asking a seller the price of a bundle of products, what would be the seller's reply?\n"
        '''The customer says, "What is the price if I add the monitor back into the deal?" \n'''
        "The seller would sell the bundle for $300, the seller replies by saying.\n"
        "<start> With the monitor included it would be $300, how does that sound? \n\n"
        "A customer is asking a seller the price of a bundle of products, what would be the seller's reply?\n"
        '''The customer says, "I am interested in this package deal, but I don't need the tripod. Can I buy it without the tripod?"\n'''
        "The seller would sell the bundle for $1500, the seller replies by saying,\n"
        "<start> Certainly, if you don't need the tripod, I can sell you the package without it for $1500. How does that sound?\n\n"
        "A customer is asking a seller the price of a product, what would be the seller's reply?\n"
        '''The customer says, "I like this speaker, I think a power cord would also be nice, what would be the price if I add it back in?"\n'''
        "The seller would sell the bundle for $200, the seller replies by saying,\n"
        "<start> The speaker along with the powercord would only cost you $200, it is a killer deal for the product.\n\n"
        "A customer is asking a seller the price of a product, what would be the seller's reply?\n"
    )

    customers_utterance = f'''The customer says, "{customers_utterance}"\n'''

    prompt_ = prompt_ + customers_utterance + \
        f"The seller would sell the bundle for ${sellers_final}, the seller replies by saying,\n"

    return gen_utterance(prompt_, temperature, max_len)


def ask_price(product_features, temperature=0.7, max_len=1000):

    prompt_ = (
        "A customer is asking a seller the price of a bundle of products.\n"
        "The bundle consists of a Dell X8 laptop, it has 16 gb ram and HD screen along with a charger.\n"
        "The customer asks, \n"
        "<start> What would be the price of the Laptop along with the charger? \n\n"
        "A customer is asking a seller the price of a bundle of products.\n"
        "The bundle consists of a Black cotton Shirt by Peter England, along with a Tie and a towel.\n"
        "The customer asks, \n"
        "<start> What would be the price for the shirt, tie and the towel? \n\n"
        "A customer is asking a seller the price of a bundle of products.\n"
        "The bundle consists of a pair of noise-cancelling headphones, a travel case, and an auxiliary cable.\n"
        "The customer asks,\n"
        "<start> What would be the total cost for the headphones, case, and cable?\n\n"
        "A customer is asking a seller the price of a bundle of products.\n"
        "The bundle consists of a 55-inch 4K Ultra HD Smart TV, a soundbar, and a wall mount.\n"
        "The customer asks,\n"
        "<start> What would be the price for the TV, soundbar, and wall mount altogether?\n\n"
        "A customer is asking a seller the price of a bundle of products.\n")

    prompt_ = prompt_ + \
        f"The bundle consists of a {product_features}.\n" + \
        "The customer asks,\n"

    return gen_utterance(prompt_, temperature, max_len)


def get_product_features(combination):

    product_feature = combination[0]
    if len(combination[1]) == 1:
        product_feature = product_feature + " and a " + combination[1][0]
    else:
        product_feature = product_feature + " along with "
        for item in combination[1][:-1]:
            product_feature = product_feature + " a " + item + ","

        product_feature = product_feature + " and a " + combination[1][-1]

    return product_feature


def main():
    buyer_intent = {
        -1: 'NA',
        0: 'Greet-Ask',
        1: 'Negotiate-Price-Decrease',
        2: 'Negotiate-Remove-X',
        3: 'Ask_Clarification-Y',
        4: 'Negotiate-Price-NoChange',
        5: 'Negotiate-Add-X',
        6: 'Ask_Price',
        7: 'Accept',
        8: 'Reject',
        9: 'Greet-Ask_Negotiate-Price-Decrease',
        10: 'Negotiate-Remove-X_Negotiate-Price-Decrease',
        11: 'Negotiate-Remove-delivery',
        12: 'tell_price'
    }
    seller_intent = {
        -1: 'NA',
        0: 'Greet-Inform',
        1: 'Negotiate-Price-NoChange',
        2: 'Negotiate-Price-Increase',
        3: 'Ask_Price',
        4: 'Provide_Clarification-Y',
        5: 'Negotiate-Add-X',
        6: 'Accept',
        7: 'Greet-Inform_Negotiate-Price-Increase',
        8: 'Acknowledge acceptance',
        9: 'Greet-Inform_Negotiate-Price-NoChange',
        10: 'Negotiate-Remove-delivery',
        11: 'Negotiate-Price-Remove-X',
        12: 'avoid_rejection'
    }
    access_list = []
    access_cost = {}
    json_file = open('product_data.json')
    data = json.load(json_file)
    for key1 in data.keys():
        for key2 in data[key1]['accessories'].keys():
            access_list.append(key2)
    for ele in access_list:
        access_cost[ele] = random.randint(500, 2000)

    for i in range(0, 4000):
        dialogue_set = get_random_flow(data,
                                       access_cost,
                                       buyer_intent=buyer_intent,
                                       seller_intent=seller_intent)
        t_flow = random.choice(dialogue_set)
        flow = parse_flow_temp(t_flow, data)

        customer_utterances = [""]
        customer_prices = [str(int(flow[-1]))]
        seller_utterances = [""]
        seller_prices = [str(int(flow[-2]))]
        utterances = [""]

        f.write(str(flow))
        f.write("\n########\n")

        for cnt, item in enumerate(flow[:-2]):

            if item[2] != 'NA':
                str_price = str(int(item[2]))
            else:
                str_price = 'NA'

            product_features = get_product_features(item[4])
            utterance = call_function_for_string(
                s=item[1],
                turn=cnt + 1,
                product_features=product_features,
                seller_initial=seller_prices[-1],
                seller_final=str_price,
                customer_initial=customer_prices[-1],
                customer_final=str_price,
                customer_utterance=customer_utterances[-1],
                seller_utterance=seller_utterances[-1],
                clarification_feature=item[7],
                removal=item[9],
                addition=item[8])

            utterances.append(utterance)

            if (cnt + 1) % 2 == 0:
                # Seller
                seller_utterances.append(utterance)
                if str_price != 'NA':
                    seller_prices.append(str_price)

            else:
                # Buyer
                customer_utterances.append(utterance)
                if str_price != 'NA':
                    # Caveat: If this price corresponds to different bundle it may be different
                    customer_prices.append(str_price)
        f.write("------------------------------------\n\n")
        # f1.write("@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n")


main()
print("DOne!")
f.close()
# f1.close()
