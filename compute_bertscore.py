# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:01:15 2020

@author: areej

code compute the similarity between references and multiple candidates using BERTScore
"""

from argparse import ArgumentParser
from bert_score import BERTScorer
import numpy as np

def get_bert_score(scorer, multi_preds, multi_golds):

    P, R, F1=[],[],[]
    for ind in range(len(multi_preds)):
        preds = multi_preds[ind]
        golds= multi_golds[ind]
        #print("golds", g)

        f,p,r=None,None,None
        for pp in preds:
            P_temp, R_temp, F1_temp = scorer.score([pp], [golds])
    
            if f is None:
                f= F1_temp
                r= R_temp
                p=P_temp
            elif F1_temp > f:
                f= F1_temp
                r= R_temp
                p= P_temp
            #print('-'*50)
        #print("choose: ", p, r, f)
        P.append(p)
        R.append(r)
        F1.append(f) 
        #print('_'*50)  
    return np.asarray(P), np.asarray(R), np.asarray(F1)


# args
parser = ArgumentParser()
# general params
parser.add_argument("-g", "--gold", required=True)
parser.add_argument("-p", "--pred", required=True)
parser.add_argument("-l", "--lang",default='eng')


args = parser.parse_args()



gold_path=args.gold
pred_path=args.pred

l= args.lang



with open(pred_path) as f:  
    try: 
        preds= [line.split(',') for line in f]
        preds =[[s.strip() for s in l] for l in preds]
    except:
        preds= [line.strip().split(',') for line in f]
        
        

with open(gold_path) as f:
    try: 
        golds= [line.split(',') for line in f]
        golds =[[s.strip() for s in l] for l in golds]
    except:
        golds=[line.strip().split(',') for line in f]



scorer = BERTScorer(lang='en')

if 'tfidf' in pred_path.split('/')[-1]:
    print("in tfidf == true")
    P, R, F= get_bert_score(scorer, preds, golds)
else:
    P, R, F= get_bert_score(scorer, preds, golds)
    #P, R, F= scorer.score(cands, refs)
print(f"P={P.mean().item():f} R={R.mean().item():f} F={F.mean().item():f}")

