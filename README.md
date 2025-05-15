# CL_TeamLab: Setting up the Project Guide
### Contributors:    
- [Pema Gurung](https://github.com/pemagrg1)   
- [Selman Aydin](https://github.com/selman-aydin)   
- [Wen Wen](https://github.com/WenW186079)   

<b>Task: Document Classification: Age Classification</b>

## 1. Install Python 3.8  
[Download link](https://www.python.org/downloads/)

## 2. Setup Virtual Env
- Step1: Install  ``` pip install virtualenv ``` 
- Step2: Run the command ```python<version> -m venv <virtual-environment-name> ``` Example: ``` python -m venv my_venv ```
- Step3: Activate the Virtual env ```source my_venv/bin/activate```
<br>Note: We will be installing all the packages to <b>my_venv</b> environment

## 3. Install the packages
List of requirements: 
```
python=3.8  
pandas>=1.3.0  
numpy>=1.21.0  
scikit-learn>=0.24.0
matplotlib
python-dotenv
```
- To install the required dependencies in your environment run: 
```
bash install.sh
```

## 4. Setup the .env file
Create a .env(to set the folder and file paths) file inside the project folder.<br>
<b>Note:</b>Please keep the variable name same as /CL_TeamLab/env_example.txt 

## 5. Train the baseline model
Use the following code to train the model.
```
python modules/main_base.py
```

## 6. Evaluation of the baseline model
- Dummy Classifier: [dummy_classifier.py](modules/dummy_classifier.py)
- Model Evaluation: [evaluation.py](modules/evaluation.py)

## 7. Advance Model
- [Fine tuning Llama2](https://github.com/pemagrg1/CL_TeamLab/blob/main/Advanced_Method/finetune_llama2_model.py)
- [Fine tuning Llama3](https://github.com/pemagrg1/CL_TeamLab/blob/main/Advanced_Method/Fine_tune_llama3.py)

## 8. Evaluation of Advance Model
- [Evaluation](https://github.com/pemagrg1/CL_TeamLab/blob/main/Advanced_Method/eval_fnetuned_model.py)

## 9. Data Insights
- [Diving deeper into the data](https://github.com/pemagrg1/CL_TeamLab/blob/main/notebooks/Findings.ipynb)
-----

# Project Folder Description
## 1. data Folder
<i>Contains the train and test dataset for this Project</i> <br>
Required dataset is in [Data](https://github.com/pemagrg1/CL_TeamLab/tree/main/data)

## 2. modules Folder
<i>Contains all the modules for this Project</i> <br>

- Evaluation:
  - [Dummy Classifier Evaluation](modules/dummy_classifier.py)
  - [Evaluation](modules/evaluation.py)
  - [Gold Label to Gold Label evaluation](modules/goldlabelevaluation.py)
- [Text Preprocessing](modules/preprocess.py)
- [Tfidf](modules/tfidf_featureSelection.py) <br>
<b>Note: </b> For PCA, we have used ``` from sklearn.decomposition import PCA ``` <br>

- Baseline:
  - [Decision Tree Model](modules/decisiontree.py)
  - [Random Forest Model](modules/randomforest.py)
  - [Main Baseline file](modules/main_base.py)
- [Hyper Parameter Selection](modules/hyperparameters_selection.py)

## 3. notebooks Folder
<i> Contains the data analysis and model selection notebook </i><br>
- [DataAnalysis Notebook](notebooks/DataAnalysis.ipynb)
- [Model Selection Notebook](notebooks/model_selection.ipynb)
- [Findings Notebook](https://github.com/pemagrg1/CL_TeamLab/blob/main/notebooks/Findings.ipynb)


## 4. Advanced_Method Folder
<i> Contains all the methods used for Advance approach </i>
- [Fine tuning Llama2](https://github.com/pemagrg1/CL_TeamLab/blob/main/Advanced_Method/finetune_llama2_model.py)
- [Fine tuning Llama3](https://github.com/pemagrg1/CL_TeamLab/blob/main/Advanced_Method/Fine_tune_llama3.py)

## 5. Fine tuned model stored on drive and huggingface
- [fine tuned llama2, local](https://drive.google.com/drive/folders/11asCfVp-bWNEP7aA0Dt-hEBMqBMk8IrA?usp=sharing)
- [fine tuned llama3, local](https://drive.google.com/drive/folders/1m_Lc3BkIu0woNqaADsJn19Vz3BppOexr?usp=drive_link)
- [fine tuned llama3, hugging face](https://huggingface.co/WenWW/llama3_will_be_fine)
