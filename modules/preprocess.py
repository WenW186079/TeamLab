#!/usr/bin/env python

"""
    This file contains the preprocess functions. eg: lowercase, stopwords removal, lemmatize, stemming, punctuaction removal, etc

Task:
    Selman: Preprocessing(loading the data and Punctuation removal)
    Wen: Preprocessing(lowerCase text and number removal)
    Pema: Preprocessing(stopwords removal and tokenization)
"""
import re 
import os
import json
import pandas as pd
from collections import Counter
from dotenv import find_dotenv
from dotenv import load_dotenv


env_file = find_dotenv(".env")
load_dotenv(env_file)

class Preprocess:
    def __init__(self, stop_words_path):
        self.stop_words_path = stop_words_path
    
    def load_data(self,path):
        with open(path) as json_file:
            json_data = json.load(json_file)
            dfItem = pd.DataFrame.from_records(json_data)
            return dfItem
        
    def word_tokenize(self, text):
        """This module tokenizes the text

        Args:
            text (str): text 

        Returns:
            list: list of words
        """
        word_tokens = re.findall(r"[@#]?(?:(?:\w+(?:-|'|[.]|[.]'))+\w+|\w+)", text)
        return word_tokens
        
    def remove_stop_words(self, text): 
        """This module removes stop words

        Args:
            text (str): Input text

        Returns:
            str: cleaned text
        """
        
        with open(self.stop_words_path, "r") as f:
            stop_words = set(f.read().splitlines())

        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
        
    def lowercase_text(self, text):
        """This module converts text to lowercase

        Args:
            text (str): Input text

        Returns:
            str: Lowercased text
        """
        lowered_text = text.lower()
        return lowered_text
    
    def remove_numbers(self, text):
        """Removes numbers from the text

        Args:
            text (str): Input text

        Returns:
            str: Text with numbers removed
        """
        nonumber_text = re.sub(r'\d+', '', text)
        return nonumber_text

    def remove_punctiation(self,text):
        """Removes punctiations from the text

        Args:
            text (str): Input text

        Returns:
            str: Text with punctiations removed
        """
        return re.sub(r'[^\w\s]', '', text)
    
    def applyPreProcessing(self,text):
        text = self.lowercase_text(text)
        text = self.remove_numbers(text)
        text = self.remove_punctiation(text)
        text = self.remove_stop_words(text)
        text = self.word_tokenize(text)
        return text

    def extract_top_tokens(self, texts, labels, n=20):
        """
        Extracts the top n tokens for each label from a list of texts with corresponding labels.

        Args:
            texts (list): List of tokenized texts.
            labels (list): List of labels.
            n (int): Number of top tokens to extract for each label.

        Returns:
            dict: A dictionary where keys are labels and values are lists of top tokens.
        """
        label_tokens = {}
        for i, text in enumerate(texts):
            label = labels[i]
            if label not in label_tokens:
                label_tokens[label] = Counter()
            label_tokens[label].update(text)
        
        top_tokens = {}
        for label, counter in label_tokens.items():
            top_tokens[label] = [token for token, _ in counter.most_common(n)]
        
        return top_tokens
    
    def filter_tokens(self, texts, top_tokens):
        """
        Filters tokens in texts based on top tokens for each label.

        Args:
            texts (list): List of tokenized texts.
            top_tokens (dict): Dictionary containing top tokens for each label.

        Returns:
            list: List of filtered tokenized texts.
        """
        filtered_texts = []
        for text in texts:
            filtered_text = [token for token in text if any(token in top_tokens[label] for label in top_tokens)]
            filtered_texts.append(filtered_text)
        return filtered_texts
        
        
''' 

Example use:
preprocess = Preprocess(stop_words_path)
train_df = preprocess.load_data(train_path)
x_train = train_df['text'].apply(preprocess.applyPreProcessing)


top_tokens = preprocess.extract_top_tokens(x_train, y_train, n=30)
x_train_filtered = preprocess.filter_tokens(x_train, top_tokens)

'''
    
    
    
