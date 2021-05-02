# -*- coding: utf-8 -*-
"""bert_similarity.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZsaciMwmDMwpmO6F070-viwOVHzcQEbw

# Installing sentence-transformers and Loading raw data
"""

# !pip install -U sentence-transformers

import pandas as pd

url = 'https://raw.githubusercontent.com/kazzyabe/Bert-based_Explanation/master/bert-based_explanation/bert_data_proccessed.csv'

data = pd.read_csv(url)
data['label']=1-data['decision']

"""# Bert Embeddings and similarity"""

from sentence_transformers import SentenceTransformer,CrossEncoder,losses,util,InputExample,SentencesDataset
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn

# model2=SentenceTransformer('stsb-roberta-base')
# model=CrossEncoder('cross-encoder/stsb-roberta-base')

def cos_sim(d,model):
  emb1 = model.encode(d['query_p'])
  emb2 = model.encode(d['citation_p'])
  cos_sim =util.pytorch_cos_sim(emb1, emb2)
  return cos_sim.item()

def sts_sim(d,model):
  scores = model.predict(d)
  return scores

# data['cos_sim'] = data.apply(lambda d: cos_sim(d), axis=1)

# dlist=list(data.apply(lambda d:(d['query_p'],d['citation_p']), axis=1))
# data['sts_sim']=sts_sim(dlist)

# data['cos_sim']

# data['sts_sim']

# # yh1 = data['cos_sim']
# yh2 = data['sts_sim']
# y = data['label']
# lw=2

# fpr,tpr,thresholds=metrics.roc_curve(y,yh1,pos_label=1)
# auc=metrics.roc_auc_score(y,yh1)
# plt.plot(fpr,tpr,lw=2,label='AUC (%0.2f) '%auc)
# plt.xlabel('True Positive Rate')
# plt.ylabel('False Positive Rate')
# plt.plot([0, 1], [0, 1],color='navy',lw=lw,linestyle='--')
# plt.legend(loc="lower right")

# fpr,tpr,thresholds=metrics.roc_curve(y,yh2,pos_label=1)
# auc=metrics.roc_auc_score(y,yh2)
# plt.plot(fpr,tpr,lw=2,label='AUC (%0.2f) '%auc)
# plt.xlabel('True Positive Rate')
# plt.ylabel('False Positive Rate')
# plt.plot([0, 1], [0, 1],color='navy',lw=lw,linestyle='--')
# plt.legend(loc="lower right")

# precision,recall,thresholds=metrics.precision_recall_curve(y,yh2,pos_label=1)
# avg_pr=metrics.average_precision_score(y,yh2)
# plt.plot(recall,precision,lw=lw,label='AVG_Pr (%0.2f) '%avg_pr)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.legend(loc="upper right")

# f1scores=[2*(precision[i]*recall[i])/(precision[i]+recall[i]) for i in range(len(thresholds))]
# f1=max(f1scores)
# mthres=np.round(thresholds[np.nanargmax(f1scores)],2)
# print("Max:",max(thresholds))
# print("Min:",min(thresholds))
# plt.plot(thresholds,f1scores,lw=lw,label=f'Max F1 ({np.round(f1,2)} at '+str(mthres)+')')
# plt.xlabel('Thresholds')
# plt.ylabel('F1 Scores')
# plt.legend(loc="upper right")
# print(mthres)

# yh=np.zeros(len(yh2))
# yh[yh2>=mthres]=1
# acc=metrics.accuracy_score(y, yh)
# print(acc)

"""## Downloading

# Bert Classifier
"""

# from sklearn.model_selection import train_test_split
# train,test=train_test_split(data,test_size=0.2,stratify=data['label'],random_state=1)
# # train,valid=train_test_split(train,test_size=0.2,stratify=train['label'],random_state=1)
# from torch.utils.data import DataLoader
# train_loss = losses.CosineSimilarityLoss(model)
# data_=DataLoader([InputExample(texts=[d['query_p'],d['citation_p']],label=float(d['label'])) for i,d in data.iterrows()],shuffle=True,batch_size=1)
# train_=DataLoader([InputExample(texts=[d['query_p'],d['citation_p']],label=float(d['label'])) for i,d in train.iterrows()],shuffle=True,batch_size=1)
# test_=DataLoader([InputExample(texts=[d['query_p'],d['citation_p']],label=float(d['label'])) for i,d in test.iterrows()])
# # valid_=DataLoader([InputExample(texts=[d['query_p'],d['citation_p']],label=float(d['label'])) for i,d in valid.iterrows()])
# model.fit(train_objectives=[(train_, train_loss)], epochs=5)

data = data.sample(frac=1,random_state=1).reset_index(drop=True)
skf = StratifiedKFold(n_splits=5)
splits=[(x,y) for x,y in skf.split(data, data['label'])]

fprlist=[]
tprlist=[]
auclist=[]
preclist=[]
recallist=[]
avgprlist=[]
f1scoreslist=[]
f1list=[]
acclist=[]
import torch
print(torch.cuda.is_available())
for b in [8]:
  for l in [2e-5]:
    for e in [4]:
      for train_index, test_index in splits:
        #resetting the model for every fold
        model=CrossEncoder('cross-encoder/stsb-roberta-base',num_labels=1)
        # model= SentenceTransformer('stsb-roberta-base')
        #train split
        train=data.loc[train_index]
        #test split
        test=data.loc[test_index]
        #data loaders
        train_=SentencesDataset([InputExample(texts=[d['query_p'],d['citation_p']],label=int(d['label'])) for i,d in train.iterrows()],model)
        test_=SentencesDataset([InputExample(texts=[d['query_p'],d['citation_p']],label=int(d['label'])) for i,d in test.iterrows()],model)
        train_=DataLoader(train_,batch_size=b)
        test_=DataLoader(test_)
        #loss function
        # train_loss = losses.SoftmaxLoss(model,sentence_embedding_dimension=model.get_sentence_embedding_dimension(),num_labels=2)
        #training
        # model.fit(train_objectives=[(train_, train_loss)],epochs=e,optimizer_params={'lr':l})
        model.fit(train_,epochs=e,optimizer_params={'lr':l})
        #predictions using cos_similarity
        y=test['label']
        dlist=list(test.apply(lambda d:(d['query_p'],d['citation_p']), axis=1))
        yh=sts_sim(dlist,model)
        # yh=test.apply(lambda d: cos_sim(d,model), axis=1)
        #roc
        fpr,tpr,thresholds=metrics.roc_curve(y,yh,pos_label=1)
        auc=metrics.roc_auc_score(y,yh)
        fprlist.append(fpr)
        tprlist.append(tpr)
        auclist.append(auc)
        print(auc)
        #precision recall
        precision,recall,thresholds=metrics.precision_recall_curve(y,yh,pos_label=1)
        avg_pr=metrics.average_precision_score(y,yh)
        preclist.append(precision)
        recallist.append(recall)
        avgprlist.append(avg_pr)
        print(avg_pr)
        #f1
        f1scores=[2*(precision[i]*recall[i])/(precision[i]+recall[i]) for i in range(len(thresholds)-1)]
        f1=max(f1scores)
        f1scoreslist.append(f1scores)
        f1list.append(f1)
        print(f1)
        #accuracy
        mthres=np.round(thresholds[np.nanargmax(f1scores)],2)
        yh1=np.zeros(len(yh))
        yh1[yh>=mthres]=1
        acc=metrics.accuracy_score(y, yh1)
        print(acc)
        acclist.append(acc)
      print(b,l,e)
      print("Average AUC across folds:",np.mean(auclist))
      print("Average Avg_pr across folds:",np.mean(avgprlist))
      print("Average F1 across folds:",np.mean(f1list))
      print("Average Acc across folds:",np.mean(acclist))

"""Final Results"""

# print("Average AUC across folds:",np.mean(auclist))
# print("Average Avg_pr across folds:",np.mean(avgprlist))
# print("Average F1 across folds:",np.mean(f1list))
# print("Average Acc across folds:",np.mean(acclist))