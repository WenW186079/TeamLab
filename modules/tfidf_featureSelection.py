import numpy as np

class TFIDFVectorizer:
    def __init__(self):
        pass
    
    def term_freq(self, texts):
        """Calculate the term frequency and also build the vocab along with it.

        Args:
            texts (list of list): list of row data from the dataframe. Ex: [[row1 token list],[row2 token list]]

        Returns:
            _type_: vocab, term frequency 
        """
        tf_dict = {}
        vocab = {}
        
        for text in texts:
            # unique_words = set(text)
            # for word in unique_words:
            for word in text:
                if word not in vocab:
                    vocab[word] = len(vocab)
                    tf_dict[word] = 1
                else:
                    tf_dict[word] += 1
        return tf_dict, vocab
        
    def idf(self,n_documents,tf_dict):
        """calculate the inverse document frequency
        Formula1:
            IDF<word> = No. of doc / No. of doc having <word>
        Formula2: to avoid ZeroDivisionError
            IDF<word> = No. of doc / (1+ No. of doc having <word>) 

        Args:
            n_documents (int): Total number of document
            tf_dict (dict): term frequency dictionary that contains the term and its frequency

        Returns:
            _type_: idf dict that has the computed idf score using the formula.
        """
        idf_dict = {}
        for word in tf_dict:
            # Calculate the idf score for each word
            
            # try:
            #     score = np.log(n_documents / tf_dict[word])
            # except ZeroDivisionError:
            #     score = 0.0
            # idf_dict[word] = score
            
            # Added 1 to the denominator so as to avoid ZeroDivisionError
            idf_dict[word] = np.log(n_documents / (1 + tf_dict[word]))
            
        return idf_dict
    
    def tfidf(self,n_documents, vocab,texts,idf_score):
        """calculate the tfidf = tf*idf

        Args:
            n_documents (int): No of documents
            vocab (dict): list of unique tokens
            texts (list of list): list of row data from the dataframe. Ex: [[row1 token list],[row2 token list]]
            idf_score (dict): dictionary containing idf score of each token

        Returns:
            numpy array: tfidf score of each row
        """
        # Initialize X with zeros
        X = np.zeros((n_documents, len(vocab)))
        
        for i, text in enumerate(texts):
            for word in text:
                if word in vocab:
                    X[i, vocab[word]] += 1
        
            # Normalize termfrequency by document length else scores are  89: {0.0, 1.0, 2.0, 3.0, 4.0},
            #after preprocessin, some data becomes empty so to handle that:
            if len(text) == 0:
                X[i] = np.zeros(X[i].shape)
            else:
                X[i] = X[i] / len(text)
            # X[i] = X[i] / len(text)
            
        # Apply IDF weighting
        for i in range(len(texts)):
            for j in range(len(vocab)):
                if X[i, j] > 0:
                    try:
                        X[i, j] *= idf_score[j]
                    except:
                        pass
        return X
            
    def tfidf_vector(self, texts, N=100):
        """
        Code to generate tfidf_vector for each document

        Args:
            texts (list of list): list of row data from the dataframe. Ex: [[row1 token list],[row2 token list]]
            N (int): Number of features to Select

        Returns:
            numpy array: tfidf score with Feature selection of each row
        """
        
        n_documents = len(texts)
        
        # Build the vocabulary and Compute term frequency
        tf_dict, vocab = self.term_freq(texts)
        
        # Compute IDF
        idf_score = self.idf(n_documents,tf_dict)
        
        #Compute TfIdf
        X = self.tfidf(n_documents, vocab,texts,idf_score)
        # return X #without Feature Selection
        
        # Select top-N features based on TF-IDF scores
        top_n_indices = np.argsort(np.sum(X, axis=0))[-N:]  # Select top-N indices based on sum of TF-IDF scores
        self.selected_features = {word: idx for word, idx in vocab.items() if idx in top_n_indices}

        return X[:, top_n_indices] #With Feature Selection



"""
# TEST
import pandas as pd
import json
path = "/Volumes/MacPema/Stuttgart/Sem3_Courses/Team_Lab/CL_TeamLab/data/en.train_copy.jsonl"

with open(path) as json_file:
    json_data = json.load(json_file)
    df = pd.DataFrame.from_records(json_data)
preprocess = Preprocess()
x_train = df['text'].apply(preprocess.applyPreProcessing)
top_n = 100  # Data dimension/feature selection
tfidf_vectorizer = TFIDFVectorizer()
X_tfidf_train = tfidf_vectorizer.tfidf_vector(x_train,N=top_n)
print(X_tfidf_train.shape)
non_zero = {}
for i in range(len(x_train)):
    index = i
    if set(X_tfidf_train[i]) != {0.0}:
        non_zero[index] = set(X_tfidf_train[i])
print("Non zero data:",len(non_zero))

"""