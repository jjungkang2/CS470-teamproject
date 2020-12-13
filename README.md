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
For training, you need to download processed dataset from ~.ipynb file.
You can get the dataset by running above file in google colab.
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
If main.py works well, you don't need to execute this part separately.    
However, if not, you need to read below comments and execute TextToSignLang.

### Requirements for TextToSignLang
* Python and Pakages : Python, nltk, shutil, difflib, selenium, shlex
* Other Programs : ffmpeg, Chrome driver

The requirements(some files and Other Programs) are in the 'Text_to_sign_language' folder.   
However, please download new 'Text_to_sign_language' folder from below link just in case.   
https://drive.google.com/file/d/1DLSQsq1NP_WhM7PP3h-vpkCZvRdCLKfT/view?usp=sharing
Please download this folder and use as workfolder.

### how to use
First of all, PLEASE USE ABOVE "Text_to_sign_language" FOLDER AS WORKFOLDER !  
You need to cd to above folder.   
If you not, it cannot find some path or programs.   
Then, open TextToSignLang.py and modify two PATH variables(SIGN_PATH, DOWN_PATH) as your local environment.   
If you downloaded new 'Text_to_sign_language' folder from link, you need to move 'disfluency_remover_result.txt' to this folder from 'Results' folder.    
It is because new 'Text_to_sign_language' folder has some changes in path variable.

## Limiations
### Requirements for Limitation-continuous alphabet pictures
You need to get the alphabet picture database from below link.   
https://drive.google.com/file/d/1pIdbnEjXeQqiXmrhwd6WLsKwi1l3DuJp/view?usp=sharing  

### Requirements for Limitaiton-Korean TTS
You don't need to prepare something since they are Colab file.  
However, we didn't prepare translation model. 
You need to translate processed english text(output of DisfluencyRemover) and use it or you can put the translated text as text_input_TTS.txt in your GDrive.

## Main.py
The process from applying speech to text to disfluency detection can be executed by main.py file.
Please set your work directory as 'CS470-teamproject'. 
Download some files mentioned earlier such as pre-trained model for disfluency detection and programs for generating sign language video, etc. and add to the work directory.
Note that you need to set configurations to run. Required packages what you need to install are below.
* Cython
* easydict
* ffmpy
* ibm-watson
* nltk 
* numpy
* pytorch-pretrained-bert
* torch
* transliterate

### Causion
main.py could make error because of difference of ffmpeg version(according to your OS) or cannot set up environment path.   
If these kinds of errors occur, you need to read "Texts To a Signlanguage Video" above and execute this part separately.
