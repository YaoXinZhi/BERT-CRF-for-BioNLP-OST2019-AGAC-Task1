## BERT-CRF for AGAC BioNLP OST2019-Task1

### Requirements

```
pip3 install -r requirements.txt
```

### Default Run

**Model training and evaluation**
```
python3 main.py
```
**Adjust hyperparameters**  
You can modify the model hyperparameters by editing the config.py file.  
```vi config.py```


### Data
label.txt contains all the labels involved in the data set, as well as the labels corresponding to [CLS], [SEP] and [Padding].   
train_input.txt, test_input.txt train_input.txt, test_input.txt files contain training data and test data in BIO format.
```
data/label.txt
data/train_input.txt
data/test_input.txt
```

### Evaluation
The current model evaluation uses the Conlleval.pl script.  
**Best Model:**   

 
 **Model** | **Accuracy**  | **Precision** | **Recall**    | **F1-score**
 ---- | ----- | ------  | ------    | ------ 
 **BERT+CRF**  | **0.9506** | **0.4746**| **0.5578**| **0.5128**


### Reference
1. **Conlleval.py** https://github.com/sighsmile/conlleval
2. **Conlleval.pl** https://www.clips.uantwerpen.be/conll2000/chunking/output.html
3. **AGAC OST-2019 Task**  https://sites.google.com/view/bionlp-ost19-agac-track

