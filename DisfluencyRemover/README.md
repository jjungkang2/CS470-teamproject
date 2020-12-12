Joint Disfluency Detection and Constituency Parsing
------------------------------------------------------------
A joint disfluency detection and constituency parsing model for transcribed speech based on [Neural Constituency Parsing of Speech Transcripts](https://www.aclweb.org/anthology/N19-1282) from NAACL 2019, with additional changes (e.g. self-training and ensembling) as described in [Improving Disfluency Detection by Self-Training a Self-Attentive Model](https://www.aclweb.org/anthology/2020.acl-main.346/) from ACL 2020.

### Task
Disfluency refers to any interruptions in the normal flow of speech, including filled pauses (*um*, *uh*), repetitions (*they're ... they're here*), corrections (*to Sydney ... no to Melbourne*), false starts (*we would like ... let's go*), parenthetical asides (*you know*, *I mean*), partial words (*wou-*, *oper-*) and interjections (*well*, *like*). One type of disfluency which is especially problematic for conventional syntactic parsers are speech repairs. A speech repair consists of three main parts; the *reparandum*, the *interregnum* and the *repair*. As illustrated in the example below, the reparandum *we don't* is the part of the utterance that is replaced or repaired, the interregnum *uh I mean* (which consists of a filled pause *uh* and a discourse marker *I mean*) is an optional part of the disfluency, and the repair *a lot of states don't* replaces the reparandum. The fluent version is obtained by removing the reparandum and the interregnum.

<p align="center">
  <img src="img/flat-ex.jpg" width=370 height=120>
</p>

This repository includes the code used for training a joint disfluency detection and constituency parsing model of transcribed speech on the Penn Treebank-3 Switchboard corpus. Since the Switchboard trees include both syntactic constituency nodes and disfluency nodes, training a parser to predict the Switchboard trees can be regarded as multi-task learning (where the tasks are syntactic parsing and identifying disfluencies). In the Switchboard treebank corpus the *reparanda*, *filled pauses* and *discourse markers* are dominated by *EDITED*, *INTJ* and *PRN* nodes, respectively. Filled pauses and discourse markers belong to a finite set of words and phrases, so INTJ and PRN nodes are trivial to detect. Detecting EDITED nodes, however, is challenging and is the main focus of disfluency detection models.

<p align="center">
  <img src="img/tree-ex.jpg" width=550 height=300>
</p>