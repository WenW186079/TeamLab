import json
import os
import sys
import pandas as pd

from modules.evaluation import Evaluation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

class GoldLabelEvaluation:
    def __init__(self):
        self.label_list = ['100s', '10s', '110s', '20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s']

    def each_label_accuracy(self, df_train):
        evaluation_dict = {}
        for each_label in self.label_list:
            gold_labels = df_train["age"]  
            evaluation_class = Evaluation(gold_labels, gold_labels, each_label)  
            evaluation_dict[each_label] = evaluation_class.f1()
            
        return evaluation_dict
