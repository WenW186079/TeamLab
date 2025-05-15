'''
Change the path if needed.

'''

import os
# Set the cache path
os.environ['TRANSFORMERS_CACHE'] = '/mount/studenten/team-lab-cl/data2024/w/data/'
print(os.getenv('TRANSFORMERS_CACHE'))

# Set the Hugging Face cache directory
os.environ['HF_HOME'] = '/mount/studenten/team-lab-cl/data2024/w/data/'


import os
import random
import functools
import csv
import pandas as pd
import numpy as np
import json
import torch
import torch.nn.functional as F
import evaluate

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score

from datasets import Dataset, DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

import torch
torch.cuda.empty_cache()

'''

Path: change the path if needed

'''
train_path = './en.train_copy.jsonl'
test_path = './en.test_copy.jsonl'


'''
For the model name, there are two approach to load the fine tuned model

1. Download the model 'saved_model' to the disk, model_name = "/path to/saved_model"

2. Directly use from hugging face, model_name = "WenWW/llama3_will_be_fine"

Note: 
These two models are different. 
As the CUDA limited, "WenWW/llama3_will_be_fine" used smaller parameters which lead poor performance.
In the paper, the result is from model_name = "/path to/saved_model"


'''

model_name = "/mount/studenten/team-lab-cl/data2024/w/data/saved_model"



# Uesful functions
def get_performance_metrics(df_test):
  y_test = df_test.age
  y_pred = df_test.predictions

  print("Confusion Matrix:")
  print(confusion_matrix(y_test, y_pred))

  print("\nClassification Report:")
  print(classification_report(y_test, y_pred))

  print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_pred))
  print("Accuracy Score:", accuracy_score(y_test, y_pred))

def load_data(path):
    with open(path) as json_file:
      json_data = json.load(json_file)
      dfItem = pd.DataFrame.from_records(json_data)
    dfItem = dfItem.drop(['gender','id','age_exact'], axis=1)
    return dfItem

df_train = load_data(train_path)
df_test = load_data(test_path)

df_train['age']=df_train['age'].astype('category')
df_train['target']=df_train['age'].cat.codes

df_test['age']=df_test['age'].astype('category')
df_test['target']=df_test['age'].cat.codes

df_train['age'].cat.categories
category_map = {code: category for code, category in enumerate(df_train['age'].cat.categories)}
category_map

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(df_train.drop('age',axis=1))
test_dataset = Dataset.from_pandas(df_test.drop('age',axis=1))
print("Convert to Hugging Face Dataset--Done!")

# Split the training data to create a validation set
train_test_split = train_dataset.train_test_split(test_size=0.2)
train_data = train_test_split['train']
val_data = train_test_split['test']
print("Split the training data to create a validation set--Done!")

# Combine into a DatasetDict
dataset = DatasetDict({
    'train': train_data,
    'val': val_data,
    'test': test_dataset
})

print(dataset)


# Balance the data
df_train.target.value_counts(normalize=True)

class_weights=(1/df_train.target.value_counts(normalize=True).sort_index()).tolist()
class_weights=torch.tensor(class_weights)
class_weights=class_weights/class_weights.sum()
class_weights

# Prepare llamla model

quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, # enable 4-bit quantization
    bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
    bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
    bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
)

lora_config = LoraConfig(
    r = 16, # the dimension of the low-rank matrices
    lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05, # dropout probability of the LoRA layers
    bias = 'none', # wether to train bias weights, set to 'none' for attention layers
    task_type = 'SEQ_CLS'
)


num_labels=11

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    num_labels=num_labels
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Prepare tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1


# Evaluation on un-fine-tuned llama3
sentences = df_test['text'].tolist()

batch_size = 32

all_outputs = []

# Process the sentences in batches
for i in range(0, len(sentences), batch_size):
    # Get the batch of sentences
    batch_sentences = sentences[i:i + batch_size]

    # Tokenize the batch
    inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Move tensors to the device where the model is (e.g., GPU or CPU)
    inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}

    # Perform inference and store the logits
    with torch.no_grad():
        outputs = model(**inputs)
        all_outputs.append(outputs['logits'])

final_outputs = torch.cat(all_outputs, dim=0)
final_outputs.argmax(axis=1)


df_test['predictions']=final_outputs.argmax(axis=1).cpu().numpy()
df_test['predictions'].value_counts()
df_test['predictions']=df_test['predictions'].apply(lambda l:category_map[l])

get_performance_metrics(df_test)
