#!/usr/bin/env python
# ~* coding: utf-8 *~
#===============================================================================
#
#           FILE: my_chatbot.py 
#         AUTHOR: Bianca Ciobanica
#	       EMAIL: bianca.ciobanica@student.uclouvain.be
#
#           BUGS: 
#        VERSION: 3.11.4
#        CREATED: 09-12-2023 
#
#===============================================================================
#    DESCRIPTION:  
#    
#          USAGE: python my_chatbot.py
#===============================================================================
import time
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

# initialize model
my_model = AutoModelForQuestionAnswering.from_pretrained("./model")
my_tokenizer = AutoTokenizer.from_pretrained("./model")

# predict answer
def predict_answer(question, context):
    # tokenize input
    inputs = my_tokenizer(question, context, return_tensors="pt")

    with torch.no_grad():
        outputs = my_model.forward(**inputs)

    # get scores
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # get highest probable answer
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits) + 1

    # get answer from indexes
    predict_answer_tokens = inputs.input_ids[0, start_index : end_index]

    answer = my_tokenizer.decode(predict_answer_tokens)

    return answer

def print_delay(text):
    for letter in text:
        print(letter, end='', flush=True)
        time.sleep(0.01)

# display welcome text
intro = "Welcome " + '\U0001F60A' + " !\n" + "This is your personal assistant that can answer a question given a context.\n" +\
"The context is a piece of text that contains the information needed to answer the question." + \
"To use the model, you need to pass the question and the context to the model's prediction function.\n\nPress 'quit' to stop it.\n\n"

print_delay(intro)

example = """Here is an example usage :\nQuestion : "Who is the author of the book 'To Kill a Mockingbird'?" \n""" + \
"""Context : "'To Kill a Mockingbird' is a novel by the American author Harper Lee. Published in 1960, it was immediately successful,""" + \
""" winning the Pulitzer Prize, and has become a classic of modern American literature. The plot and characters are loosely based on """ + \
"""Lee's observations of her family, her neighbors and an event that occurred near her hometown of Monroeville, Alabama, in 1936, """ + \
"""when she was 10 years old." \n\nAnswer : Harper Lee \n """

print_delay(example)

start = input("\nDo you want to try it out ? Enter 'yes' or 'quit' : ")
bye = "See you soon ! \nTerminating instance.\n"
empty_error = "You must provide the required fields.\n"

if start == "yes":
    is_active = True
    while is_active:
        print_delay("\nWhat is your question ? \n")
        question = input("Question: ")

        if question == "":
            print_delay(empty_error)
            continue

        if question == 'quit':
            print_delay(bye)
            break
 
        context = input("Context: ")

        if context == "":
            print_delay(empty_error)
            continue 

        answer = predict_answer(question, context)

        print_delay("Answer :\n\n" + answer + "\n")


else:
    print_delay(bye)
