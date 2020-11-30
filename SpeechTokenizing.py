from nltk.tokenize import sent_tokenize, WordPunctTokenizer

from nltk.parse.stanford import StanfordParser
from nltk.stem import WordNetLemmatizer
from nltk.tree import *

para = "this ERR over the X of V means the classification error on the data using the perturbation V so in this scenario we have an only single perturbation and when we apply this single perturbation to the datas of X than ERR is the classification error"
sents = sent_tokenize(para)
print(sents)

word_tokenize = WordPunctTokenizer().tokenize
for sent in sents:
    print(word_tokenize(sent))

    
parser = StanfordParser()

# Generates all possible parse trees sort by probability for the sentence
possible_parse_tree_list = [tree for tree in parser.parse(para.split())]