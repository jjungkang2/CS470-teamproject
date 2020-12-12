CS470-teamproject
--------------------------------------------

## Requirements

### Install packages  

### IBM Speech to Text API
```
pip install --upgrade "ibm-watson>=4.7.1"
```
You need to authenticate to the API by using IBM Cloud Identity and Access Management(IAM).
You can easily create the key in [IAM authentication](https://cloud.ibm.com/docs/watson?topic=watson-about).   

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