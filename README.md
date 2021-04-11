## BERT-CRF for BioNLP-OST2019 AGAC-Task1

### Requirements

```
pip3 install -r requirements.txt
```

### Default Run

**Model training and evaluation**
```
python3 main.py
```
**modify hyperparameters**  
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
The current model evaluation uses the Conlleval.pl script. You can view the details of the model evaluation results through logging/conlleval.log
**Best Model: (On test set)**   

 
 **Model** | **Accuracy**  | **Precision** | **Recall**    | **F1-score**
 ---- | ----- | ------  | ------    | ------ 
 **BERT+CRF**  | **95.7730** | **54.6274**| **56.4596**| **55.5284**


### Reference
1. **Conlleval.py** https://github.com/sighsmile/conlleval
2. **Conlleval.pl** https://www.clips.uantwerpen.be/conll2000/chunking/output.html
3. **BioNLP OST-2019 AGAC Task**  https://sites.google.com/view/bionlp-ost19-agac-track  
4. Wang, Yuxing, et al. **"An Overview of the Active Gene Annotation Corpus and the BioNLP OST 2019 AGAC Track Tasks."** Proceedings of The 5th Workshop on BioNLP Open Shared Tasks. 2019.
