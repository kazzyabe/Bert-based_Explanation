import os
import time
from logging import basicConfig, INFO
from smart_open import smart_open
from gensim.models import word2vec
from gensim.models import doc2vec
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

class MySentences(object):
    @staticmethod
    def preprocess(line):
        line = line.lower()
        lst = line.split()
        #lst_1 = [x for x in lst if len(str(x)) > 2 ] 
        #lst_n = [item for item in lst_1 if item.isalpha()]
        lst = [x.replace('.','') for x in lst]
        lst = [x.replace(',','') for x in lst]        
        stops = set(stopwords.words("english"))
        lst_n = [word for word in lst if word not in stops] 
        return lst_n

    def __init__(self, dirname, encoding='ISO-8859-15'):
        self.dirname = dirname
        self.encoding = encoding
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname), encoding=self.encoding):
                yield MySentences.preprocess(line)


def train(dirname):
    sentences = MySentences(dirname)

    basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', 
        level=INFO
    )

    # model = word2vec.Word2Vec( sentences, 
    #             size=4, sg=1, window=1, min_count=50, 
    #             max_vocab_size=None, workers=2, iter=1, 
    #             negative=2, trim_rule=None, sorted_vocab=1, 
    #             compute_loss=True
    #         )

    model = word2vec.Word2Vec( sentences, 
                size=300, sg=1, window=21, min_count=50, 
                max_vocab_size=None, workers=64, iter=10, 
                negative=10, trim_rule=None, sorted_vocab=1, 
                compute_loss=True
            )

    return model

def read_corpus(fname, tokens_only=False):
    with smart_open(fname, encoding='ISO-8859-15') as f:
        for i, line in enumerate(f):
            preprocess = simple_preprocess(line)
            if tokens_only:
                yield preprocess
            else:
                yield doc2vec.TaggedDocument(preprocess, [i])

if __name__ == "__main__":
    # first commands time stamps
    print(time.strftime("%Y%m%d-%H%M%S"))

    # train model
    w2v_model = train('./work')


   # save model
    w2v_model.save('/home/ubuntu/W2V_openC_AWS')


    # ----- WQ START -----
    # read files
    queries = {
        '1':['./query/Q1.txt'],
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

    # read data from query doc
    a_WQ = open('/home/ubuntu/W2V-GS-openC-300w21-10-WQ'+time.strftime("%Y%m%d-%H%M%S")+'.txt','w')
    print('Query\tCited\tDOC SIM\n',file=a_WQ)

    for q in queries:
        query = list(read_corpus(queries[q][0], tokens_only=True))
        Q = query[0]
        for n in Q:
            n.lower()
        Q = [x.replace('.','') for x in Q]
        Q = [x.replace(',','') for x in Q]
        stops = set(stopwords.words("english"))
        Q = [word for word in Q if word not in stops] 
    
        # read data from cited docs
        citeds = {
            '1':['./cited/C'+q+' 1.txt'],
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

            distance = w2v_model.wmdistance(Q, C)
            print('Q',q,'\t','C',q,' ',key,'\t','distance = %.3f' % distance,'\n',file=a_WQ)

    # close file
    a_WQ.close()
    # ----- WQ  END  -----

    # last commands time stamps
    print(time.strftime("%Y%m%d-%H%M%S"))
