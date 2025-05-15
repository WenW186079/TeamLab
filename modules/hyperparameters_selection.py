import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
import itertools
from dotenv import find_dotenv,load_dotenv

from preprocess import Preprocess
from evaluation import Evaluation
from tfidf_featureSelection import TFIDFVectorizer


env_file = find_dotenv(".env")
load_dotenv(env_file)

train_path = os.environ.get("train_path")
test_path = os.environ.get("test_path")
stop_words_path = os.environ.get("stop_words_path")
'''

This file is used to select the optimal hyperparameters using libraries.

'''

labels = {'100s':9, '10s':0, '110s':10, '20s':1, '30s':2, '40s':3, '50s':4, '60s':5, '70s':6, '80s':7, '90s':8}

def convertLabel(x):
    return labels[x]

preprocess = Preprocess(stop_words_path)
train_df = preprocess.load_data(train_path)
test_df = preprocess.load_data(test_path)

# Define hyperparameters
top_tokens_numbers = [5, 10, 15, 20, 30, 50]
top_ns = [50, 100, 150, 500]
n_components_list = [20, 50, 100]
max_depths = [10, 20, 30, 50]
num_trees_list = [50]
random_seed = 42

best_accuracy = 0
best_hyperparameters = {}

# Perform grid search
for top_tokens_number, top_n, n_components, max_depth, num_trees in itertools.product(top_tokens_numbers, top_ns, n_components_list, max_depths, num_trees_list):
    print(f"Testing hyperparameters: top_tokens_number={top_tokens_number}, top_n={top_n}, n_components={n_components}, max_depth={max_depth}, num_trees={num_trees}")
    
    # Preprocess data
    top_tokens = preprocess.extract_top_tokens(train_df['text'], train_df['age'], n=top_tokens_number)
    x_train_filtered = preprocess.filter_tokens(train_df['text'], top_tokens)
    x_test_filtered = preprocess.filter_tokens(test_df['text'], top_tokens)
    
    # Tfidf embedding
    tfidf_vectorizer = TFIDFVectorizer()
    X_tfidf_train = tfidf_vectorizer.tfidf_vector(x_train_filtered, N=top_n)
    X_tfidf_test = tfidf_vectorizer.tfidf_vector(x_test_filtered, N=top_n)
    
    # Perform PCA to reduce dimensionality
    min_samples_features = min(X_tfidf_train.shape)
    n_components = min(n_components, min_samples_features)
    if n_components < 1:
        n_components = 1
    
    pca = PCA(n_components=n_components, random_state=random_seed)
    X_tfidf_train_pca = pca.fit_transform(X_tfidf_train)
    X_tfidf_test_pca = pca.transform(X_tfidf_test)
    
    # Train the Random Forest model
    random_forest = RandomForestClassifier(n_estimators=num_trees, random_state=random_seed, max_depth=max_depth)
    random_forest.fit(X_tfidf_train_pca, train_df['age'].apply(convertLabel))
    
    # Evaluate the Random Forest model
    forest_predictions = random_forest.predict(X_tfidf_test_pca)
    accuracy = accuracy_score(test_df['age'].apply(convertLabel), forest_predictions)
    print(f"Accuracy: {accuracy}")
    
    # Update best hyperparameters if the current model is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_hyperparameters = {
            'top_tokens_number': top_tokens_number,
            'top_n': top_n,
            'n_components': n_components,
            'max_depth': max_depth,
            'num_trees': num_trees
        }

print("Best hyperparameters:", best_hyperparameters)
print("Best accuracy:", best_accuracy)
