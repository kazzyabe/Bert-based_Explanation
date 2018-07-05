import gensim.downloader as api
import gensim
import os
import collections
import smart_open
import random
import time
import numpy as np
from scipy import spatial
from nltk.corpus import stopwords

print(time.strftime("%Y%m%d-%H%M%S"))

#****************** PARAMETERS 
#the parameters for training the model are fed to gensim.models.doc2vec function

#*********************** FUNCTIONS
def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])
#***************************************************





#***************** download model *********************

w2v_gnews_model = api.load('word2vec-google-news-300')

#********************************************************


#s1_afv = avg_feature_vector('this is a sentence', model=model, num_features=300, index2word_set=index2word_set)
#s2_afv = avg_feature_vector('this is also sentence', model=model, num_features=300, index2word_set=index2word_set)
#sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
#print(sim)



##*****************read files 
#
##************************* CREATE LOOP
#
queries  =  {'1':['./query/Q1.txt'],  
                 '2':['./query/Q2.txt'],
                 '3':['./query/Q3.txt'],
                 '4':['./query/Q4.txt'],
                 '5':['./query/Q5.txt'],
                 '6':['./query/Q6.txt'],
                 '7':['./query/Q7.txt'],			 
                 '8':['./query/Q8.txt'],
                 '9':['./query/Q9.txt'],
                 '10':['./query/Q10.txt']			 
                }

##**************** READ DATA FROM QUERY DOC
a = open('PV-DM-doc-sim'+time.strftime("%Y%m%d-%H%M%S")+'.txt','w')
print('Query\tCited\tDOC SIM\n',file=a)
#counter = 0
for q in queries:
    query = list(read_corpus(queries[q][0], tokens_only=True))
    Q = query[0]
    for n in Q:
        n.lower()
    Q = [x.replace('.','') for x in Q]
    Q = [x.replace(',','') for x in Q]
    stops = set(stopwords.words("english"))
    Q = [word for word in Q if word not in stops]

##**************** READ DATA FROM CITED DOCS 

    citeds = {'1':['./cited/C'+q+' 1.txt'],  
                   '2':['./cited/C'+q+' 2.txt'],
                   '3':['./cited/C'+q+' 3.txt'],
                   '4':['./cited/C'+q+' 4.txt'],
                   '5':['./cited/C'+q+' 5.txt'],
                   '6':['./cited/C'+q+' 6.txt'],
                   '7':['./cited/C'+q+' 7.txt'],			 
                   '8':['./cited/C'+q+' 8.txt'],
                   '9':['./cited/C'+q+' 9.txt'],
                   '10':['./cited/C'+q+' 10.txt']			 
                              }
    for key in citeds:
        cited = list(read_corpus(citeds[key][0], tokens_only=True))
        C = cited[0]	
        for c in C:
            c.lower()
        C = [x.replace('.','') for x in C]
        C = [x.replace(',','') for x in C]
        stops = set(stopwords.words("english"))
        C = [word for word in C if word not in stops]


        distance = w2v_gnews_model.wmdistance(Q, C)
        print('Q',q,'\t','C',q,' ',key,'\t','distance = %.3f' % distance,'\n',file=a)


a.close()

# first and last commands are time stamps
print(time.strftime("%Y%m%d-%H%M%S"))
