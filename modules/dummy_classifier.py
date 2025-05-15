import random
import os 
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from evaluation import Evaluation

class DummyClassifier:
    def __init__(self,df_train):
        self.df_train = df_train
        self.label_list = list(set(self.df_train["age"].values))

    def assign_labels(self):
        return random.choice(self.label_list)
    
    def each_label_accuracy(self):
        self.df_train["dummy_age"] = self.df_train.apply(lambda x: self.assign_labels(), axis=1)
        
        total = 0
        evaluation_dict = {}
        # For f1 Score
#         for each_label in self.label_list:
#             evaluation_class = Evaluation(self.df_train["dummy_age"],self.df_train["age"],each_label) #predicted,goldData,class_label
#             evaluation_dict[each_label] = evaluation_class.f1_score()
        
        # For Accuracy
        for each_label in self.label_list:
            new_df_train = self.df_train[self.df_train["age"]==each_label]
            evaluation_class = Evaluation(new_df_train["dummy_age"],new_df_train["age"],each_label)
            evaluation_dict[each_label] = evaluation_class.accuracy()

            total = total+evaluation_dict[each_label]
            
        avg_score = total/len(evaluation_dict)
            
        return avg_score, evaluation_dict
 
  
#####################TEST##############################   
# import json                                         #
# import pandas as pd                                 #
# from dotenv import find_dotenv                      #
# from dotenv import load_dotenv                      #

# env_file = find_dotenv(".env")                      #
# load_dotenv(env_file)                               #
# train_data_path = os.environ.get("train_path")      #
# with open(train_data_path) as json_file:            # 
#     json_data = json.load(json_file)                # 

# df_train = pd.DataFrame.from_records(json_data)     # 
 
# d = DummyClassifier(df_train)                       # 

# avg_score, evaluation_dict = d.each_label_accuracy()# 
# print(avg_score)                                    # 
# print(evaluation_dict)                              # 
######################################################   


