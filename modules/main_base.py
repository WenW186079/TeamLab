import os
import json
import pandas as pd
import numpy as np
import random
from sklearn.decomposition import PCA
from collections import Counter
from dotenv import find_dotenv,load_dotenv

from preprocess import Preprocess
from evaluation import Evaluation
from tfidf_featureSelection import TFIDFVectorizer
from decisiontree import TreeNode,DecisionTree
from randomforest import RandomForest
from dummy_classifier import DummyClassifier

env_file = find_dotenv(".env")
load_dotenv(env_file)

train_path = os.environ.get("train_path")
test_path = os.environ.get("test_path")
stop_words_path = os.environ.get("stop_words_path")

'''
Setup all the hyperparameters

'''
top_tokens_number = 50 # Top Tokens of Each Lable
top_n = 150 # Top-N Features of TF-IDF
n_components = 20 # PCA Components 
max_depth = 20 # Max Depth of Tree
n_estimators = 50 # Num of Trees
random_seed = 42 # Freeze the seed for Randome Forest and PCA


labels = {'100s':9, '10s':0, '110s':10, '20s':1, '30s':2, '40s':3, '50s':4, '60s':5, '70s':6, '80s':7, '90s':8}
def convertLabel(x):
  return labels[x]

# Load data
preprocess = Preprocess(stop_words_path)
train_df = preprocess.load_data(train_path)
test_df = preprocess.load_data(test_path)
#print(train_df[:5])

# Dummy classification 
d = DummyClassifier(train_df)
avg_score, evaluation_dict = d.each_label_accuracy()
print("Average Dummy Accuracy:", avg_score)
print("Dummy Evaluation Dictionary:")
print(evaluation_dict)

# Preprocess data
x_train = train_df['text'].apply(preprocess.applyPreProcessing)
x_test = test_df['text'].apply(preprocess.applyPreProcessing)
#print(x_train[:5])

# Preprocess the label
y_train = train_df["age"].apply(convertLabel)
y_test = test_df["age"].apply(convertLabel)
#print(y_train[:3])

top_tokens_number = top_tokens_number
print('top_tokens_number:',top_tokens_number)
top_tokens = preprocess.extract_top_tokens(x_train, y_train, n=top_tokens_number)
'''
for label, tokens in top_tokens.items():
    print(f"Top {top_tokens_number} tokens for label '{label}':")
    print(tokens)
    print()
'''
# Filter tokens based on top tokens
x_train_filtered = preprocess.filter_tokens(x_train, top_tokens)
x_test_filtered = preprocess.filter_tokens(x_test, top_tokens)
#print(x_train_filtered[:5])

# Ifidf embedding
top_n = top_n
print("Select top-N features:", top_n)
tfidf_vectorizer = TFIDFVectorizer()
X_tfidf_train = tfidf_vectorizer.tfidf_vector(x_train_filtered,N=top_n)
X_tfidf_test = tfidf_vectorizer.tfidf_vector(x_test_filtered,N=top_n)
#print('X_tfidf',X_tfidf_train[:1])

# Perform PCA to reduce dimensionality
n_components = n_components
print('number of principal components to retain:', n_components )
pca = PCA(n_components=n_components, random_state=random_seed) # Freeze the PCA seed
#pca = PCA(n_components=n_components)
x_train_pca = pca.fit_transform(X_tfidf_train)
x_test_pca = pca.fit_transform(X_tfidf_test)
#print('x_train_PCA',x_train_pca[:1])

# Train the Decision Tree model
print('Start training tree...')
max_depth = max_depth
print("max_depth",max_depth)
decision_tree = DecisionTree(max_depth=max_depth)
decision_tree.train(x_train_pca, y_train)
print('End training tree!')

# Tree Predictions on the test set
tree_predictions = decision_tree.predict(x_test_pca)
print('tree_predictions',tree_predictions[:10])
print('y_test',y_test[:10])

tree_evaluation = Evaluation(tree_predictions, y_test)
tree_evaluation.evaluate_multiclass_metrics('Decision Tree')

# Train the Random Forest model
print('Start training forest...')
max_depth = max_depth
n_estimators = n_estimators
random_seed = random_seed
print("number of trees",n_estimators)
print("max_depth",max_depth)
random_forest = RandomForest(n_estimators=n_estimators,  random_seed = random_seed, max_depth=max_depth)
random_forest.train(x_train_pca, y_train)
print('End training forest!')

# Forest Predictions on the test set
forest_predictions = random_forest.predict(x_test_pca)
print('forest_predictions',forest_predictions[:10])
print('y_test',y_test[:10])

forest_evaluation = Evaluation(forest_predictions, y_test)
forest_evaluation.evaluate_multiclass_metrics('Random Forest')
