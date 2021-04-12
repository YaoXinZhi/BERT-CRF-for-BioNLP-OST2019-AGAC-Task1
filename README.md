## BERT-CRF for BioNLP-OST2019 AGAC-Task1  


### Cite  
Please cite this follow work, if you use this code:  
Wang, Yuxing, et al. **"An Overview of the Active Gene Annotation Corpus and the BioNLP OST 2019 AGAC Track Tasks."** Proceedings of The 5th Workshop on BioNLP Open Shared Tasks. 2019.


### Virtual Environment
You can build a virtual environment for project operation.  
```
# Building a virtual environment
pip3 install virtualenv
pip3 install virtualenvwrapper

virtualenv -p /usr/local/bin/python3.6 $env_name --clear  

# active venv.
source $env_name/bin/activate  

# deactive venv.
deactivate
```


### Requirements

```
pip3 install -r requirements.txt
```
If you cannot download torch automatically through requirements.txt, you can delete the torch version information and get the command line of torch installation from the [torch official website](https://pytorch.org/). Note that the installed torch version needs to be the same as that in requirenemts.txt.

**OSX**  
```
pip3 install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
```

**Linux and Windos**  
```
# CUDA 11.0
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.2
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2

# CUDA 10.1
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 9.2
pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
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
