# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:54:03 2021

@author: Cian
"""
#-------------------------------------------------
#           ABOUT THIS FILE
# This file is dedicated to testing the functionality of NLTK
# 
# Useful links:
# https://www.nltk.org/
# https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
#
#-------------------------------------------------


# To download the prequisites, run this once. 
def nltk_downloader():
  import nltk
  nltk.download('punkt')


def clean_text(text):
    # Create the lemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    
    # Get rid of non alpha characters except "'" as it is needed for the lemment
    text = "".join(c for c in text if c.isalnum() or c == " " or "'")
    
    # Get rid of capitals
    text = text.lower()
    
    # Tokenize the words    
    # Create tokens of each word
    token_text = word_tokenize(text)
    
    # Get rid of any piece of text that isn't over 2 characters
    token_text = [t for t in token_text if len(t) > 2] 
    
    # Put words in base form by doing lemmatization
    token_text = [wordnet_lemmatizer.lemmatize(t) for t in token_text]
    
    # Return the tokens
    return token_text


def main():
    # Download prequisites
    # nltk_downloader()

    

# Run Main
main()
