#!/usr/bin/env python
# ~* coding: utf-8 *~
#===============================================================================
#
#           FILE: google_news_exercice.py 
#         AUTHOR: Bianca Ciobanica
#	       EMAIL: bianca.ciobanica@student.uclouvain.be
#
#           BUGS: 
#        VERSION: 3.11.4
#        CREATED: 18-11-2023 
#
#===============================================================================
#    DESCRIPTION:  
#    
#          USAGE: python google_news_exercice.py 
#===============================================================================
import gensim.downloader
google_news_corpus = 'word2vec-google-news-300'
# load corpus
print("Loading corpus...")

glove_vectors = gensim.downloader.load(google_news_corpus)

print(u'\u2713' + " Corpus succesfully loaded") 


def get_5_closest_words(word):
    similarities = {result[0] for result in glove_vectors.most_similar(word)[:5]}
    print("5 closest words to", word + ":")
    print(similarities)

    return similarities

print("Calculating similar words")
car_5_closest = get_5_closest_words('car')
feature_5_closest = get_5_closest_words('feature')
computer_5_closest = get_5_closest_words('computer')



