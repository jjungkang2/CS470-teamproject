CS470-teamproject
--------------------------------------------
## Contents
* Speech To Text
* Punctuation Restoration
* DisfluencyRemover
* Texts To a Signlanguage Video
* Limiations
* Main.py

## Speech to Text
### IBM Speech to Text API
```
pip install --upgrade "ibm-watson>=4.7.1"
```
You need to authenticate to the API by using IBM Cloud Identity and Access Management(IAM).
You can easily create the key in [IAM authentication](https://cloud.ibm.com/docs/watson?topic=watson-about).   

## Punctuator Restoration
### Model Training
You can train the model from scratch, with dataset construction and model building.
Training code is provided in PunctuationRestoration/punctuator_training.ipynb file, and can be run in google colab.
Detailed instructions are provided in the file.

### Model Applying
You can apply the model with our pre-made model, datasets, dictionaries and matrices.


## DisfluencyRemover
### Requirements for DisfluencyRemover 
* Python, Cython, PyTorch, pytorch-pretrained-bert

### Pretrained Models for DisfluencyRemover
Download pretrained model and bert with following link and unzip at `DisfluencyRemover\model` folder.  
The pretrained model's f-score=92.4%.

```
$ cd model
$ wget https://github.com/pariajm/joint-disfluency-detection-and-parsing/releases/download/naacl2019/swbd_fisher_bert_Edev.0.9078.pt
$ wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz
$ tar -xf bert-base-uncased.tar.gz
```

## Texts To a Signlanguage Video
### Requirements for TextToSignLang
* Python and Pakages : Python, nltk, shutil, difflib, selenium, shlex
* Other Programs : ffmpeg, Chrome driver

The requirements with Other Programs are in the folder at below link.   
https://drive.google.com/file/d/1ygMqnPPBsBTiJar3WxOC-n2NyShIOq5N/view?usp=sharing  
Please download this folder and use as workfolder.

### how to use
First of all, PLEASE USE ABOVE LINKED FOLDER AS WORKFOLDER !  
You need to cd to above folder.   
If you not, it cannot find some path or programs.   
Then, open TextToSignLang.py and modify two PATH variables(SIGN_PATH, DOWN_PATH) as your local environment.

## Limiations
### Requirements for Limitation-continuous alphabet pictures
You need to get the alphabet picture database from below link   
https://drive.google.com/file/d/1pIdbnEjXeQqiXmrhwd6WLsKwi1l3DuJp/view?usp=sharing  

### Requirements for Limitaiton-Korean TTS
You don't need to prepare something since they are Colab file.  
However, we didn't prepare translation model. 
You need to translate processed english text(output of DisfluencyRemover) and use it or you can put the translated text as text_input_TTS.txt in your GDrive.

## Main.py
### down link : 
