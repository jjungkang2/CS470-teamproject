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


### DisfluencyRemover - Requirements for Training
* Python 3.6 or higher.
* Cython 0.25.2 or any compatible version.
* [PyTorch](http://pytorch.org/) 0.4.1, 1.0/1.1, or any compatible version.
* [pytorch-pretrained-bert](https://github.com/huggingface/pytorch-pretrained-BERT) 0.4.0 or any compatible version


### DisfluencyRemover - Pretrained Models
The following pre-trained models, which have been optimized for their performance on parsing EDITED nodes i.e. F(S_E) on the SWBD dev set, are available for download:
* [`swbd_fisher_bert_Edev.0.9078.pt`](https://github.com/pariajm/joint-disfluency-detector-and-parser/releases/download/naacl2019/swbd_fisher_bert_Edev.0.9078.pt): Model self-trained on the Switchboard gold parse trees and Fisher silver parse trees with BERT-base-uncased word representations (EDITED word f-score=92.4%).

```
$ cd models
$ wget https://github.com/pariajm/joint-disfluency-detection-and-parsing/releases/download/naacl2019/swbd_fisher_bert_Edev.0.9078.pt
$ wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
$ wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz
$ tar -xf bert-base-uncased.tar.gz && cd ..
$ python main.py
```