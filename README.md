# Overview
This repository implements a sarcasm detection model by utilizing a 
sentence pair classification BERT architecture.

# Setup
You must
* Clone the bert git repo not pip install it (does not contain latest updates)
* preferably conda install tensorflow-gpu 1.15.0 (or a stable setup process you know works)
* install tensorflow_hub=0.7.0  

You could try ```conda env create -f environment.yml```  

# Data Shape and Form
Sarcasm Data from twitter and reddit have the following attributes
* label : one of 'SARCASM', 'NOT_SARCASM'
* response: the response to be classified as Sarcastic, given context
* context: A list of upto 3 context items [c1, c2, c3] with c3 preceding the response 

# Run the Model
```python
from preprocess import load_data, standard_loader
from model import SarcasmBertBasic
cfg = {
    'model_ver': 'v1',
    'BERT_MODEL': 'uncased_L-12_H-768_A-12',
    'NUM_TRAIN_EPOCHS': 3.0,
}

train, test = load_data(load_fn=standard_loader('both'), context_extent='all', split=0.2)
model = SarcasmBertBasic(cfg)
model.fit(train)
preds = model.predict(test)
```
Also take a look at 
```
run_model.ipynb
```  

# Results
On a twitter data set described in the paper below, we are able to achieve the following results  
[Sarcasm Analysis using Conversation Context](https://arxiv.org/abs/1808.07531)
```
Accuracy:0.833

SARCASM
Precision:0.8032786885245902
Recall:0.882
F1 Score:0.8408007626310773
    
NOT SARCASM
Precision:0.8691796008869179
Recall:0.784
F1 Score:0.8243953732912724
```
## Notes on Training
* 

# References
* Paper - [Sarcasm Analysis using Conversation Context](https://arxiv.org/abs/1808.07531)
* [Bert Repo](https://github.com/google-research/bert)
* [Official BERT Colab Notebook](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb?hl=fr)
* Another repo implementing BERT based sarcasm detection - [Sarcasm with BERT](https://github.com/blazejdolicki/bert_sarcasm_detection) 


