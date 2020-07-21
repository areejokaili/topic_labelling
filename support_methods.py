# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:24:36 2020

@author: areej
"""
import unicodedata
import re
import tensorflow as tf
from nltk.corpus import stopwords   
stop_words = set(stopwords.words('english')) 
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



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def get_optimizer(opt_choice, lr_choice, latent_dims):
        

    if opt_choice == 'rmsprop':
        print("rmsprop used")
        opt= tf.keras.optimizers.RMSprop(lr=lr_choice, rho=0.9)

    elif opt_choice == 'adam':
        print("adam used")
        opt= tf.keras.optimizers.Adam(lr=lr_choice, beta_1=0.9, beta_2=0.999, amsgrad=True)
        
    else:
        print("optimizer name not found")
        sys.exit(0)
    
    return opt


# Converts the unicode file to ascii
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
  w = unicode_to_ascii(w.lower().strip())

  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

  w = w.rstrip().strip()

  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
  w = 'sostok ' + w + ' eostok'
  return w

def max_length(tensor):
  return max(len(t) for t in tensor)

def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))


def seq2text(lang, tensor):
  newString=''
  for t in tensor:
    if t!=0:
      newString+=lang.index_word[t]+' '
  return newString

def text_cleaner(text,num):
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
        if len(i)>0:  #removing short word (KEEP ALL)
            long_words.append(i)   
    return (" ".join(long_words)).strip()



def preprocess_using_tokenizer(data, x_tokenizer, y_tokenizer, max_text_len):    
        
    cleaned_terms = []
    # split to max size before adding the end/start tokens

    for t in data['terms']:
        #print("before clean", t)
        t = ' '.join(t.split()[:max_text_len-2])
        cleaned_terms.append(text_cleaner(t,0))  
        #print('after clean', cleaned_terms[len(cleaned_terms)-1])
    ## call the function
    cleaned_labels = []
    for t in data['labels']:
        #print("t:", t)
        cleaned_labels.append(t)   # don't  need to clean y in test
    data['cleaned_terms']=cleaned_terms
    data['cleaned_labels']=cleaned_labels
    print("Test data")
    print("after cleaning text", len(data['cleaned_terms']))
    print("after cleaning summary", len(data['cleaned_labels']))
    # vmware server virtual oracle update virtualization application infrastructure management microsoft mac parallels os apple x hardware /
    #desktop allows users 
    # software machine intel hypervisor virtualized linux run product operating machines leopard
    
    
    data['cleaned_terms']  = data['cleaned_terms'].apply(lambda x : 'sostok '+ x + ' eostok')
    data['cleaned_labels'] = data['cleaned_labels'].apply(lambda x : 'sostok '+ x + ' eostok')
    
    x_test_seq = x_tokenizer.texts_to_sequences(data['cleaned_terms'])

    #y_test_seq = y_tokenizer.texts_to_sequences(data['cleaned_labels']) 
    #print("data['cleaned_labels'][0]", data['cleaned_labels'][0], " --> to sequence -->", y_test_seq[0])

    y =data['labels'].values.tolist()
    #print("y[0]", y[0])
    return x_test_seq,  y



def preprocess_y(y):
    y_after_append=[]
    for row in y:
        labels= row.split(',')
        labels_append=''
        for label in labels:
            label= 'sostok '+ label + ' eostok'
            labels_append+=label +' '
        y_after_append.append(labels_append)
    return y_after_append



