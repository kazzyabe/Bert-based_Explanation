from __future__ import print_function
import gensim
import os
import collections
import smart_open
import random
import time


print(time.strftime("%Y%m%d-%H%M%S"))

##
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

#******************** LOAD TRAINING DATA
train_corpus = list(read_corpus('data.txt'))
#Let's take a look at the training corpus
#print('\n 2 first in the train corpus',train_corpus[:2])
#print('\nthe train corpus',train_corpus)

#Change from test to actual vector_size 300 epochs = 500 min_count = 5
#******************instantiates the object Doc2Vec
model = gensim.models.doc2vec.Doc2Vec(vector_size=300,window=21,min_count=5, epochs=5000)


#***************** Build a Vocabulary
model.build_vocab(train_corpus)
#print('\nprinting vocabulary',model.wv.vocab)

#*****************Train model
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
print('\nmodel trained')
model.save("d2vACLgensimmodel.d2v")


#*************read files 


            #************************* LIST OF FILES
file_pairs = {1:['./8Q-cited-txt/C0101.txt','./8Q-txt/Q01-4000.txt'],
              2:['./8Q-cited-txt/C0102.txt','./8Q-txt/Q01-5000.txt'],
              3:['./8Q-cited-txt/C0103.txt','./8Q-txt/Q01-1000.txt'],
              4:['./8Q-cited-txt/C0104.txt','./8Q-txt/Q01-7000.txt'],
              5:['./8Q-cited-txt/C0105.txt','./8Q-txt/Q01-2000.txt'],
              6:['./8Q-cited-txt/C0106.txt','./8Q-txt/Q01-1000.txt'],
              7:['./8Q-cited-txt/C0107.txt','./8Q-txt/Q01-3000.txt'],
              8:['./8Q-cited-txt/C0108.txt','./8Q-txt/Q01-1000.txt'],
              9:['./8Q-cited-txt/C0109.txt','./8Q-txt/Q01-1000.txt'],
              10:['./8Q-cited-txt/C0110.txt','./8Q-txt/Q01-4000.txt'],
              11:['./8Q-cited-txt/C0201.txt','./8Q-txt/Q02-3000.txt'],
              12:['./8Q-cited-txt/C0202.txt','./8Q-txt/Q02-2000.txt'],
              13:['./8Q-cited-txt/C0203.txt','./8Q-txt/Q02-1000.txt'],
              14:['./8Q-cited-txt/C0204.txt','./8Q-txt/Q02-2000.txt'],
              15:['./8Q-cited-txt/C0205.txt','./8Q-txt/Q02-8000.txt'],
              16:['./8Q-cited-txt/C0206.txt','./8Q-txt/Q02-1000.txt'],
              17:['./8Q-cited-txt/C0207.txt','./8Q-txt/Q02-1000.txt'],
              18:['./8Q-cited-txt/C0208.txt','./8Q-txt/Q02-1000.txt'],
              19:['./8Q-cited-txt/C0209.txt','./8Q-txt/Q02-1000.txt'],
              20:['./8Q-cited-txt/C0210.txt','./8Q-txt/Q02-1000.txt'],
              21:['./8Q-cited-txt/C0301.txt','./8Q-txt/Q03-1000.txt'],
              22:['./8Q-cited-txt/C0302.txt','./8Q-txt/Q03-2000.txt'],
              23:['./8Q-cited-txt/C0303.txt','./8Q-txt/Q03-5000.txt'],
              24:['./8Q-cited-txt/C0304.txt','./8Q-txt/Q03-2000.txt'],
              25:['./8Q-cited-txt/C0305.txt','./8Q-txt/Q03-1000.txt'],
              26:['./8Q-cited-txt/C0306.txt','./8Q-txt/Q03-8000.txt'],
              27:['./8Q-cited-txt/C0307.txt','./8Q-txt/Q03-1000.txt'],
              28:['./8Q-cited-txt/C0308.txt','./8Q-txt/Q03-2000.txt'],
              29:['./8Q-cited-txt/C0309.txt','./8Q-txt/Q03-2000.txt'],
              30:['./8Q-cited-txt/C0310.txt','./8Q-txt/Q03-5000.txt'],
              31:['./8Q-cited-txt/C0401.txt','./8Q-txt/Q04-7000.txt'],
              32:['./8Q-cited-txt/C0402.txt','./8Q-txt/Q04-2000.txt'],
              33:['./8Q-cited-txt/C0403.txt','./8Q-txt/Q04-2000.txt'],
              34:['./8Q-cited-txt/C0404.txt','./8Q-txt/Q04-2000.txt'],
              35:['./8Q-cited-txt/C0405.txt','./8Q-txt/Q04-2000.txt'],
              36:['./8Q-cited-txt/C0406.txt','./8Q-txt/Q04-1000.txt'],
              37:['./8Q-cited-txt/C0407.txt','./8Q-txt/Q04-7000.txt'],
              38:['./8Q-cited-txt/C0408.txt','./8Q-txt/Q04-7000.txt'],
              39:['./8Q-cited-txt/C0409.txt','./8Q-txt/Q04-2000.txt'],
              40:['./8Q-cited-txt/C0410.txt','./8Q-txt/Q04-2000.txt'],
              41:['./8Q-cited-txt/C0501.txt','./8Q-txt/Q05-2000.txt'],
              42:['./8Q-cited-txt/C0502.txt','./8Q-txt/Q05-2000.txt'],
              43:['./8Q-cited-txt/C0503.txt','./8Q-txt/Q05-2000.txt'],
              44:['./8Q-cited-txt/C0504.txt','./8Q-txt/Q05-2000.txt'],
              45:['./8Q-cited-txt/C0505.txt','./8Q-txt/Q05-4000.txt'],
              46:['./8Q-cited-txt/C0506.txt','./8Q-txt/Q05-2000.txt'],
              47:['./8Q-cited-txt/C0507.txt','./8Q-txt/Q05-4000.txt'],
              48:['./8Q-cited-txt/C0508.txt','./8Q-txt/Q05-2000.txt'],
              49:['./8Q-cited-txt/C0509.txt','./8Q-txt/Q05-2000.txt'],
              50:['./8Q-cited-txt/C0510.txt','./8Q-txt/Q05-2000.txt'],
              51:['./8Q-cited-txt/C0601.txt','./8Q-txt/Q06-6000.txt'],
              52:['./8Q-cited-txt/C0602.txt','./8Q-txt/Q06-1000.txt'],
              53:['./8Q-cited-txt/C0603.txt','./8Q-txt/Q06-1000.txt'],
              54:['./8Q-cited-txt/C0604.txt','./8Q-txt/Q06-2000.txt'],
              55:['./8Q-cited-txt/C0605.txt','./8Q-txt/Q06-4000.txt'],
              56:['./8Q-cited-txt/C0606.txt','./8Q-txt/Q06-1000.txt'],
              57:['./8Q-cited-txt/C0607.txt','./8Q-txt/Q06-2000.txt'],
              58:['./8Q-cited-txt/C0608.txt','./8Q-txt/Q06-1000.txt'],
              59:['./8Q-cited-txt/C0609.txt','./8Q-txt/Q06-2000.txt'],
              60:['./8Q-cited-txt/C0610.txt','./8Q-txt/Q06-1000.txt'],
              61:['./8Q-cited-txt/C0701.txt','./8Q-txt/Q07-2000.txt'],
              62:['./8Q-cited-txt/C0702.txt','./8Q-txt/Q07-2000.txt'],
              63:['./8Q-cited-txt/C0703.txt','./8Q-txt/Q07-1000.txt'],
              64:['./8Q-cited-txt/C0704.txt','./8Q-txt/Q07-2000.txt'],
              65:['./8Q-cited-txt/C0705.txt','./8Q-txt/Q07-2000.txt'],
              66:['./8Q-cited-txt/C0706.txt','./8Q-txt/Q07-2000.txt'],
              67:['./8Q-cited-txt/C0707.txt','./8Q-txt/Q07-2000.txt'],
              68:['./8Q-cited-txt/C0708.txt','./8Q-txt/Q07-1000.txt'],
              69:['./8Q-cited-txt/C0709.txt','./8Q-txt/Q07-2000.txt'],
              70:['./8Q-cited-txt/C0710.txt','./8Q-txt/Q07-1000.txt'],
              71:['./8Q-cited-txt/C0801.txt','./8Q-txt/Q08-1000.txt'],
              72:['./8Q-cited-txt/C0802.txt','./8Q-txt/Q08-8000.txt'],
              73:['./8Q-cited-txt/C0803.txt','./8Q-txt/Q08-7000.txt'],
              74:['./8Q-cited-txt/C0804.txt','./8Q-txt/Q08-8000.txt'],
              75:['./8Q-cited-txt/C0805.txt','./8Q-txt/Q08-8000.txt'],
              76:['./8Q-cited-txt/C0806.txt','./8Q-txt/Q08-7000.txt'],
              77:['./8Q-cited-txt/C0807.txt','./8Q-txt/Q08-1000.txt'],
              78:['./8Q-cited-txt/C0808.txt','./8Q-txt/Q08-7000.txt'],
              79:['./8Q-cited-txt/C0809.txt','./8Q-txt/Q08-8000.txt'],
              80:['./8Q-cited-txt/C0810.txt','./8Q-txt/Q08-4000.txt'],
              81:['./8Q-cited-txt/C0901.txt','./8Q-txt/Q09-2000.txt'],
              82:['./8Q-cited-txt/C0902.txt','./8Q-txt/Q09-2000.txt'],
              83:['./8Q-cited-txt/C0903.txt','./8Q-txt/Q09-2000.txt'],
              84:['./8Q-cited-txt/C0904.txt','./8Q-txt/Q09-1000.txt'],
              85:['./8Q-cited-txt/C0905.txt','./8Q-txt/Q09-2000.txt'],
              86:['./8Q-cited-txt/C0906.txt','./8Q-txt/Q09-2000.txt'],
              87:['./8Q-cited-txt/C0907.txt','./8Q-txt/Q09-2000.txt'],
              88:['./8Q-cited-txt/C0908.txt','./8Q-txt/Q09-2000.txt'],
              89:['./8Q-cited-txt/C0909.txt','./8Q-txt/Q09-2000.txt'],
              90:['./8Q-cited-txt/C0910.txt','./8Q-txt/Q09-2000.txt'],
              91:['./8Q-cited-txt/C1001.txt','./8Q-txt/Q10-1000.txt'],
              92:['./8Q-cited-txt/C1002.txt','./8Q-txt/Q10-7000.txt'],
              93:['./8Q-cited-txt/C1003.txt','./8Q-txt/Q10-3000.txt'],
              94:['./8Q-cited-txt/C1004.txt','./8Q-txt/Q10-8000.txt'],
              95:['./8Q-cited-txt/C1005.txt','./8Q-txt/Q10-7000.txt'],
              96:['./8Q-cited-txt/C1006.txt','./8Q-txt/Q10-7000.txt'],
              97:['./8Q-cited-txt/C1007.txt','./8Q-txt/Q10-7000.txt'],
              98:['./8Q-cited-txt/C1008.txt','./8Q-txt/Q10-1000.txt'],
              99:['./8Q-cited-txt/C1009.txt','./8Q-txt/Q10-2000.txt'],
              100:['./8Q-cited-txt/C1010.txt','./8Q-txt/Q10-1000.txt']}

#**************** READ DATA FROM QUERY DOC
a = open('PVDM-300-21-5000'+time.strftime("%Y%m%d-%H%M%S")+'.txt','w')
print('Query','\t','Cited','\t','DOC SIM\n',file=a)
for qc in file_pairs:
    query = list(read_corpus(file_pairs[qc][1], tokens_only=True))
    Q = query[0]

#**************** READ DATA FROM CITED DOCS 


    cited = list(read_corpus(file_pairs[qc][0], tokens_only=True))
    C = cited[0]	
#*****************Computing similarity
    sim = model.docvecs.similarity_unseen_docs(model,Q, C)
    print('Q',file_pairs[qc][1],'\t','C',file_pairs[qc][0],' ','\t',sim,'\n',file=a)

a.close()


print(time.strftime("%Y%m%d-%H%M%S"))
