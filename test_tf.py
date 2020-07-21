# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:11:32 2019

@author: areej
"""
import os
import sys
import support_methods
import pickle
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
import pandas as pd
import model_archi_tf as model_archi
import support_methods as sup_methods
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns; sns.set()



def remove_tokens(sent, flag):
    if flag:
        return sent.replace('eostok sostok',',').replace('sostok','').replace('eostok','')
    else:
        return sent.replace('sostok','').replace('eostok','')
    
    
    
def evaluate(sentence):
  attention_plot = np.zeros((max_summary_len, max_text_len))
  if 'sostok' not in sentence:
    sentence = support_methods.preprocess_sentence(sentence)

  inputs = [x_tokenizer.word_index[i] for i in sentence.split()]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_text_len,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = [tf.zeros((1, latent_dim))]
  #print("initial hidden shape", len(hidden), len(hidden[0]), len(hidden[0][0]))
  enc_out, enc_hidden = encoder(inputs, hidden)
  
  #print('enc_hidden shape', enc_hidden.shape)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([y_tokenizer.word_index['sostok']], 0)

  for t in range(max_summary_len):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)
    #print('attention weights:', attention_weights)
    # storing the attention weights to plot later on
    if np.all(attention_weights) != None: 
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()

    result += y_tokenizer.index_word[predicted_id] + ' '

    if y_tokenizer.index_word[predicted_id] == 'eostok':
      return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot


# function for plotting the attention weights
def plot_attention(attention, predicted_sentence, sentence , num):
    plt.figure(figsize=(10, 4))
 
    ax=sns.heatmap(attention, xticklabels=sentence, yticklabels=predicted_sentence, cmap="OrRd")
    ax.xaxis.set_ticks_position('top')

    ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
    #plt.show()
    plt.draw()
    
    if num <= 10:
          plt.savefig('results/attention_'+str(num)+'.png', bbox_inches = 'tight')


  
def predict(sentence, num):
  result, sentence, attention_plot = evaluate(sentence)
  
  if not np.all(attention_plot == 0) :
      a = result.strip().split(' ')
      b = sentence.strip().split(' ')
      attention_plot = attention_plot[:len(a), :len(b)]
      plot_attention(attention_plot, a, b, num)
  return result, sentence


    
''' End of Functions '''


  
parser = ArgumentParser()
parser.add_argument("-d", "--data", required=True, help="data_folder: wiki_tfidf or wiki_sent")
parser.add_argument("-m", "--model_name", required=True,  help="model name (bigru_bahdanau_attention)")
parser.add_argument("-ld", "--latent_dim", default=200, type=int, help="Latented dim")
parser.add_argument("-ed", "--embed_dim", default=300,type=int, help="Embedding dim")
parser.add_argument("-opt", "--optimizer", default='adam', help="optimizer algorithm (default adam)")
parser.add_argument("-lr", "--learning_rate",default=0.001, type=float, help="learning rate (default 0.001)")

parser.add_argument("-s", "--sub_set",type=int, help="data subset")
parser.add_argument("-bs", "--batch_size",default=128, type=int, help="batch size (default 128)")
parser.add_argument("-p", "--print",default='False')
parser.add_argument( "--load", required=True, help="load specific model name")
parser.add_argument("-te", "--topic_evaluate", default=False , help="test on different data (deafult False) "\
                    " - write bhatia_topics to test on topic terms on it own"\
                    " - write bhatia_topics_tfidf to test on topics+additional terms ")

args = parser.parse_args()


                        
# info 
data_name= args.data
model_name=args.model_name
topic_data= args.topic_evaluate
print_flag = args.print
################
# model parameters
latent_dim=args.latent_dim
embedding_dim=args.embed_dim
BATCH_SIZE = args.batch_size
''' 
Predicting 
Load needed files 
'''

# loading X tokenizer
with open('data/'+data_name+'/x_tokenizer.pickle', 'rb') as handle:
    x_tokenizer= pickle.load(handle)
print('number of x_tokenizer words --->', x_tokenizer.num_words)


# loading Y tokenizer
with open('data/'+data_name+'/y_tokenizer.pickle', 'rb') as handle:
    y_tokenizer= pickle.load(handle)    
print('number of y_tokenizer words --->', y_tokenizer.num_words)


#size of vocabulary ( +1 for padding token)
x_voc   =  x_tokenizer.num_words + 1 #use num_words instead of word_index since we have a large vocab words and would like to keep the top frequent only 
vocab_inp_size = x_voc
#size of vocabulary
y_voc  =   y_tokenizer.num_words +1
vocab_tar_size = y_voc
print('number of vocab in x', x_voc)
print('number of vocab in y', y_voc)


'set sequence length (+2 to accommodate for eostok and sostok)'
max_text_len=30 + 2
max_summary_len=8  + 2 

    
    
if topic_data == False: # test on wiki data 
    print("load test data from", data_name)
    x_test=np.load('data/'+data_name+'/x_test.npy')
    y_test=np.load('data/'+data_name+'/y_test.npy')
    
elif topic_data != False: # for inference on different data, load raw data and preprocess using the tokenizer of the data the model was trained on
    print("load test data from", topic_data)
    data_test= pd.read_csv('data/'+topic_data+'/'+args.topic_evaluate+'.csv', names=['labels', 'terms'], header=None)
    print("first row, labels:", data_test.iloc[0], '\nterms:',data_test.iloc[0]['terms'])
    # process topic using specific tokenizer
    x_test, y_test = sup_methods.preprocess_using_tokenizer(data_test, x_tokenizer, y_tokenizer, max_text_len)  
    x_test= tf.keras.preprocessing.sequence.pad_sequences(x_test,  maxlen=max_text_len, padding='post')
    # prep y by removing ',' and adding [eostok and sostok]
    y_test = sup_methods.preprocess_y(y_test)
    
    
try:
    x_test= x_test[:args.sub_set]
    y_test= y_test[:args.sub_set]
    print("test on parts of the data, subset=", args.sub_set)
except:
    pass
      
print("testing on", len(x_test))       


if 'bigru_bahdanau_attention' == model_name:
    print("BiGRU +  attention") 
    encoder = model_archi.Encoder_bigru_attention(vocab_inp_size, embedding_dim, latent_dim, BATCH_SIZE, x_tokenizer )
    decoder = model_archi.Decoder_bigru_attention(vocab_tar_size, embedding_dim, latent_dim, BATCH_SIZE, y_tokenizer)
else:
    print("model name not found")
    sys.exit(0)
    
   

'''
Optimizer
'''
optimizer = sup_methods.get_optimizer(args.optimizer, args.learning_rate, args.latent_dim)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')




# restoring from specific checkpoint 
checkpoint_dir = './training_checkpoints/'+ data_name
print("Load specific model:", args.load)
best_model_name = args.load
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
print("restore from", checkpoint_dir+'/'+best_model_name)
checkpoint.restore((checkpoint_dir+'/'+best_model_name))




''' start testing ''' 
golds, preds, topics=[],[],[]
import time
start= time.time()
for i in range(0,len(x_test)):

    
    topic=support_methods.seq2text(x_tokenizer, x_test[i])
    if args.topic_evaluate == False:
        gold = support_methods.seq2text(y_tokenizer, y_test[i])
    else:
        gold = y_test[i]
    
    pred, topic =predict(topic, i)
        
    # remove end and start tokens
    if args.topic_evaluate != False and 'tfidf' in topic_data:
        topic = list(data_test['terms'])[i]
        topic= ' '.join(topic.split()[:10])
    topic =remove_tokens(topic, False)
    gold= remove_tokens(gold, True)
    pred = remove_tokens(pred, False)

    
    preds.append(pred)
    golds.append(gold)
    topics.append(topic)

    if i%500==0:
        end= time.time()       
        print("reached %i/%i took %.2f minutes "%(i, len(x_test),(end-start)/60))
        start = time.time()
    
#################  write predictions to file to use BERTScore 

if args.topic_evaluate == False:
    pred_path= 'results/'+ data_name+'/'+model_name+'_pred.out'
    gold_path= 'results/'+ data_name+'/'+model_name+'_gold.out'
    top_path=  'results/'+ data_name+'/'+model_name+'_topics.out'
else:
    pred_path= 'results/'+ data_name+'/'+model_name+'_'+topic_data+'_pred.out'
    gold_path= 'results/'+ data_name+'/'+model_name+'_'+topic_data+'_gold.out'
    top_path= 'results/'+ data_name+'/'+model_name+'_'+topic_data+'_topics.out'
    
# create folder if not exist
if not os.path.exists('results/'+ data_name):
    os.makedirs('results/'+ data_name)
    


'''
in bhatia_topics_tfidf the same topic has many labels we have to combine labels of the same topic to be ready for evaluation   
'''
if topic_data != False and 'tfidf' in topic_data: 
    new_preds={}
    new_golds={}
    for topic, gold, pred in zip(topics, golds, preds):
        if topic not in new_preds:
            new_preds[topic]= [pred]
            new_golds[topic] =[gold]
        else:
            new_preds[topic].append(pred)
            new_golds[topic].append(gold)
    topics= list(new_preds.keys())
    print("number of topics", len(topics))
    golds= list(new_golds.values())
    print("number of golds:", len(golds))
    preds= list(new_preds.values())
    print("number of preds:", len(preds))

with open(pred_path, 'w') as p:
    with open(gold_path, 'w') as  g:
        with open(top_path, 'w') as  t:
            for i in range (len(golds)):
                try:
                    p.write(preds[i]+'\n')
                except:
                    p.write(','.join(preds[i])+'\n')
                try:
                    g.write(golds[i]+'\n')
                except:
                    g.write(','.join(golds[i])+'\n')
                t.write(topics[i]+'\n')