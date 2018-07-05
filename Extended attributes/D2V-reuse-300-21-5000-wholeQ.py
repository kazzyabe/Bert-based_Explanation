from __future__ import print_function
import gensim
import os
import collections
import smart_open
import random
import time


print(time.strftime("%Y%m%d-%H%M%S"))

####
###****************** PARAMETERS 
###the parameters for training the model are fed to gensim.models.doc2vec function
##
###*********************** FUNCTIONS
def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

#******************instantiates the object Doc2Vec
model = gensim.models.doc2vec.Doc2Vec(vector_size=300,window=21,min_count=5, epochs=5000)


###*****************how model was trained

#***************** Build a Vocabulary
##model.build_vocab(train_corpus)
###print('\nprinting vocabulary',model.wv.vocab)
##
###*****************Train model
##model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
##print('\nmodel trained')
##model.save("d2vACLgensimmodel.d2v")
##

#*************load trained ACL model


model = gensim.models.doc2vec.Doc2Vec.load(("d2vACLgensimmodel.d2v"))

#*************read files 



#************************* CREATE LOOP

queries  =  {'1':['./wholeQ-txt/Q1.txt'],  
                 '2':['./wholeQ-txt/Q2.txt'],
                 '3':['./wholeQ-txt/Q3.txt'],
                 '4':['./wholeQ-txt/Q4.txt'],
                 '5':['./wholeQ-txt/Q5.txt'],
                 '6':['./wholeQ-txt/Q6.txt'],
                 '7':['./wholeQ-txt/Q7.txt'],			 
                 '8':['./wholeQ-txt/Q8.txt'],
                 '9':['./wholeQ-txt/Q9.txt'],
                 '10':['./wholeQ-txt/Q10.txt']			 
                }

#**************** READ DATA FROM QUERY DOC
a = open('D2V-reuse300-5-5000-wholeQ'+time.strftime("%Y%m%d-%H%M%S")+'.txt','w')
print('Query','\t','Cited','\t','DOC SIM\n',file=a)
for q in queries:
    query = list(read_corpus(queries[q][0], tokens_only=True))
    Q = query[0]
    #print(q)
#**************** READ DATA FROM CITED DOCS 

    arguments = {'1':['./wholeQ-cited-txt/C'+q+' 1.txt'],  
                   '2':['./wholeQ-cited-txt/C'+q+' 2.txt'],
                   '3':['./wholeQ-cited-txt/C'+q+' 3.txt'],
                   '4':['./wholeQ-cited-txt/C'+q+' 4.txt'],
                   '5':['./wholeQ-cited-txt/C'+q+' 5.txt'],
                   '6':['./wholeQ-cited-txt/C'+q+' 6.txt'],
                   '7':['./wholeQ-cited-txt/C'+q+' 7.txt'],			 
                   '8':['./wholeQ-cited-txt/C'+q+' 8.txt'],
                   '9':['./wholeQ-cited-txt/C'+q+' 9.txt'],
                   '10':['./wholeQ-cited-txt/C'+q+' 10.txt']			 
                               }
    for key in arguments:
        cited = list(read_corpus(arguments[key][0], tokens_only=True))
        C = cited[0]
        #print(key)
#*****************Computing similarity





#*****************Computing similarity
        sim = model.docvecs.similarity_unseen_docs(model,Q, C)
        print('Q',q,'\t','C',q,' ',key,'\t',sim,'\n',file=a)

a.close()


print(time.strftime("%Y%m%d-%H%M%S"))
