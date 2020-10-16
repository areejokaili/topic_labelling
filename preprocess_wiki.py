#!/usr/bin/env opennmt
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:08:18 2019

@author: areej


this script will pre process the data and save the data in format ready for training

parts of the preprocessing are borrowed from https://github.com/aravindpai/How-to-build-own-text-summarizer-using-deep-learning/blob/master/How_to_build_own_text_summarizer_using_deep_learning.ipynb
"""



import os
import pickle
import numpy as np  
import pandas as pd 
import re     
import csv      
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords   
import warnings
warnings.filterwarnings("ignore")


def rare_words_analysis(tokenizer, thresh):
    rare_cnt=0
    tot_cnt=0
    freq=0
    tot_freq=0
    for key,value in tokenizer.word_counts.items():
        tot_cnt=tot_cnt+1
        tot_freq=tot_freq+value
        if(value<thresh):
            rare_cnt=rare_cnt+1
            freq=freq+value
    
    print("% of rare words in vocabulary:",(rare_cnt/tot_cnt)*100)
    print("Total Coverage of rare words:",(freq/tot_freq)*100)
    return rare_cnt, tot_cnt

# wiki_title_topn_doc is the wiki data we extracted and saved in the format (title, topn words using tfidf, the first n words)
in_data_path='/Users/areej/Desktop/wiki_extract/wiki_title_topn_doc/'
in_topics_path = 'data/bhatia_topics/bhatia_topics_data_whole.csv' # we are using this to filter out any wiki titles that exactly match labels from the topics


# now processing for wiki_sent, if you want to prep for wiki_tfidf then its 'data/wiki_tfidf/'
out_data_path='data/wiki_sent/'


topics_data=pd.read_csv(in_topics_path)
topics_labels= list(topics_data['label'])


titles=[]
topns=[]
sents=[]
for subdir, dirs, files in os.walk(in_data_path):
    for file in files:
        
        
        if 'DS_Store' not in file :
            filepath = subdir + file
            print ("intput file " ,filepath)
            
            with open(filepath, 'r' , encoding='utf8') as f:
                read = csv.reader(f)
                for row in read :
                    title, topn , sent = row 
                    if title not in topics_labels:
                        #print(title, '--->',topn)
                        
                        titles.append(title)
                        topns.append(topn)
                        sents.append(sent)
                        if len(titles)> 1000:
                            break
                    

data = pd.DataFrame(
    {
     'titles': titles,
     'topns': topns,
     'sents': sents
     
    })





print("data len", len(data))
data.drop_duplicates(subset=['titles'],inplace=True)  #dropping duplicates
data.drop_duplicates(subset=['topns'],inplace=True)  #dropping duplicates
data.dropna(axis=0,inplace=True)   #dropping na
print("after removing duplicates", len(data))


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}



stop_words = set(stopwords.words('english')) 

def text_cleaner(text,num):
    #print("text before clean", text)
    newString = text.lower()
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    newString = re.sub('[m]{2,}', 'mm', newString)
    if(num==0):
        tokens = [w for w in newString.split() if not w in stop_words]
    else:
        tokens=newString.split()
    long_words=[]
    for i in tokens:
        if len(i)>1:  #removing short word
            long_words.append(i)  
    #print("text after cleaning", long_words)
   
    return (" ".join(long_words)).strip()




#start cleaning the data
# option 1 means that stopwords will not be removed
# while option 0 mean they will be removed.
    

cleaned_titles = []
for t in data['titles']:
    cleaned_titles.append(text_cleaner(t,1))
    
cleaned_topns = []
for t in data['topns']:
    cleaned_topns.append(text_cleaner(t,1))


cleaned_sents = []
for t in data['sents']:
    cleaned_sents.append(text_cleaner(t,0))

    

print("after cleaning titles", len(cleaned_titles))
print("after cleaning topns", len(cleaned_topns))
print("after cleaning sents", len(cleaned_sents))
# should be equal
assert len(cleaned_titles) == len(cleaned_topns) == len(cleaned_sents)


# these were set after examining the data distribution
max_title_len=8
max_topn_len= 30
max_sent_len= 30


# now remove any dataitem where the titles does not follow the length limit
short_titles=[]
short_topns=[]
short_sents=[]

for i in range(len(cleaned_titles)):
    if len(cleaned_titles[i].split()) <= 8:
        short_titles.append(cleaned_titles[i])
        short_topns.append(' '.join(cleaned_topns[i].split()[:max_topn_len]))
        short_sents.append(' '.join(cleaned_sents[i].split()[:max_sent_len]))
        



print('after title length filter left with', len(short_titles))


# we want to prep the wiki_sent data so from here we will continue its prep  and use "sents"
# if you want "wiki_tfidf",  then continue with topn instead if sents      
df=pd.DataFrame({'titles':short_titles,'sents':short_sents})

del short_titles
del short_topns
del short_sents

df['titles'] = df['titles'].apply(lambda x : 'sostok '+ x + ' eostok')



x_tr,x_val,y_tr,y_val=train_test_split(np.array(df['sents']),np.array(df['titles']),test_size=0.05,random_state=0,shuffle=True)
print("len(x_tr), len(x_val)",len(x_tr), len(x_val) )

x_tr,x_test,y_tr,y_test=train_test_split(x_tr,y_tr,test_size=0.05,random_state=0,shuffle=True)
print("len(x_tr), len(x_test)",len(x_tr), len(x_test) )
print("first row in train", x_tr[0])

#prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer() 
x_tokenizer.fit_on_texts(list(x_tr))


thresh=4
rare_cnt, tot_cnt = rare_words_analysis(x_tokenizer, thresh)

#prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer(num_words=tot_cnt-rare_cnt) 
x_tokenizer.fit_on_texts(list(x_tr))

#convert text sequences into integer sequences
x_tr_seq    =   x_tokenizer.texts_to_sequences(x_tr) 
x_val_seq   =   x_tokenizer.texts_to_sequences(x_val)
x_test_seq   =   x_tokenizer.texts_to_sequences(x_test)

#padding zero upto maximum length
x_tr    =   pad_sequences(x_tr_seq,  maxlen=max_sent_len, padding='post')
x_val   =   pad_sequences(x_val_seq, maxlen=max_sent_len, padding='post')
x_test   =   pad_sequences(x_test_seq, maxlen=max_sent_len, padding='post')

#size of vocabulary ( +1 for padding token)
x_voc   =  x_tokenizer.num_words + 1


y_tokenizer = Tokenizer()   
y_tokenizer.fit_on_texts(list(y_tr))


thresh=6
rare_cnt, tot_cnt= rare_words_analysis(y_tokenizer, thresh)



#prepare a tokenizer for reviews on training data

y_tokenizer = Tokenizer(num_words=tot_cnt-rare_cnt) 
y_tokenizer.fit_on_texts(list(y_tr))
print("y_tokenizer number of words", y_tokenizer.num_words)

#convert text sequences into integer sequences
y_tr_seq    =   y_tokenizer.texts_to_sequences(y_tr) 
y_val_seq   =   y_tokenizer.texts_to_sequences(y_val) 
y_test_seq   =   y_tokenizer.texts_to_sequences(y_test) 

#padding zero upto maximum length
y_tr    =   pad_sequences(y_tr_seq, maxlen=max_title_len+2, padding='post')
y_val   =   pad_sequences(y_val_seq, maxlen=max_title_len+2, padding='post')
y_test   =  pad_sequences(y_test_seq, maxlen=max_title_len+2, padding='post')

#size of vocabulary
y_voc  =   y_tokenizer.num_words +1

# now deleting the rows that contain only sostok and eostok tokens
ind=[]
for i in range(len(y_tr)):
    cnt=0
    for j in y_tr[i]:
        if j!=0:
            cnt=cnt+1
    if(cnt==2):
        ind.append(i)

y_tr=np.delete(y_tr,ind, axis=0)
x_tr=np.delete(x_tr,ind, axis=0)


ind=[]
for i in range(len(y_val)):
    cnt=0
    for j in y_val[i]:
        if j!=0:
            cnt=cnt+1
    if(cnt==2):
        ind.append(i)

y_val=np.delete(y_val,ind, axis=0)
x_val=np.delete(x_val,ind, axis=0)

ind=[]
for i in range(len(y_test)):
    cnt=0
    for j in y_test[i]:
        if j!=0:
            cnt=cnt+1
    if(cnt==2):
        ind.append(i)

y_test=np.delete(y_test,ind, axis=0)
x_test=np.delete(x_test,ind, axis=0)

print("after deleting len(x_tr) len(x_val) len(x_test)",len(x_tr) ,len(x_val) ,len(x_test))
# save data
np.save(out_data_path +'y_tr.npy', y_tr)    # .npy extension is added if not given
np.save(out_data_path + 'x_tr.npy', x_tr)    

np.save(out_data_path +'y_val.npy', y_val)   
np.save(out_data_path +'x_val.npy', x_val)     

np.save(out_data_path +'y_test.npy', y_test)    
np.save(out_data_path +'x_test.npy', x_test)    


#save tokemizer
# dump X tokenizer
with open(out_data_path +'x_tokenizer.pickle', 'wb') as handle:
   pickle.dump(x_tokenizer,handle)


# dump Y tokenizer
with open(out_data_path +'y_tokenizer.pickle', 'wb') as handle:
   pickle.dump( y_tokenizer, handle)
    

