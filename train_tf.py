# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:29:54 2019

@author: areej areej.okaili@sheffield.ac.uk, or okaili.areej@gmail.com


parts of code based on 
https://www.tensorflow.org/
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


import os
import sys
import time
import numpy as np
import random as rn
# set random seed for reproducibility
rn.seed(1)
np.random.seed(1)


from argparse import ArgumentParser
import pickle
from sklearn.utils import shuffle
import model_archi_tf as model_archi
import support_methods as sup_methods
import warnings
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



'''
Test if GPU is used or not
'''
print('#'*15)
print("Is gpu available?", tf.test.is_gpu_available())
print("Is gpu built with cuda?",tf.test.is_built_with_cuda())

try:
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
        print("**calculation on GPU\n", c)
except:
    print("**CPU is used")

print('#'*15)




def loss_function(real, pred):
  #mask out index with padding value= 0, because it doesn't participate in the prediction process
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


def validate(inputs, targ, enc_hidden):
  loss = 0
  #print("inp shape", inp.shape, ' enc_hidden shape', enc_hidden.shape)
  enc_output, enc_hidden = encoder(inp, enc_hidden)
  dec_hidden = enc_hidden    # passing enc_output to the decoder

  dec_input = tf.expand_dims([y_tokenizer.word_index['sostok']] * BATCH_SIZE, 1)
  for t in range(1, targ.shape[1]):
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
    
      loss += loss_function(targ[:, t], predictions)
      dec_input = tf.expand_dims(targ[:, t], 1) 

  batch_loss = (loss / int(targ.shape[1]))
  return batch_loss


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


def get_optimizer(opt_choice, lr_choice, latent_dims,schedule_flag):

    if schedule_flag:
        print("initilize schedulling...")
        learning_rate = CustomSchedule(latent_dims)
        
        if opt_choice == 'rmsprop':
            print('with rmsprop')
            opt= tf.keras.optimizers.RMSprop(learning_rate , rho=0.9)
            return opt
        elif opt_choice == 'adam':
            print('with adam')
            opt= tf.keras.optimizers.Adam(learning_rate , beta_1=0.9, beta_2=0.999, amsgrad=True)
            return opt
        

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


def get_epoch_num(name):
    name= name.split('_')
    print(name)
    loc=0
    epoch_loc=None
    for n in name:
        if n =='e':
            epoch_loc= loc+1
            break 
        else:
            loc+=1
    
    return int(name[epoch_loc])
        
            

parser = ArgumentParser()
# general params
parser.add_argument("-d", "--data", required=True, help="data_folder wiki_tfidf or wiki_sent")
parser.add_argument("-m", "--model_name", required=True,  help="model name (bigur_bahdanau_attention)")
parser.add_argument("-s", "--sub_set",type=int, help="data subset")
parser.add_argument("-bs", "--batch_size",default=128, type=int, help="batch size (default 128)")
parser.add_argument("-save", "--save_model",default=True, help="save model (default True)")
parser.add_argument("--checkpoint",default='', help="checkpoint name")


# neural params
parser.add_argument("-ld", "--latent_dim", default=200, type=int, help="Latented dim")
parser.add_argument("-ed", "--embed_dim", default=300,type=int, help="Embedding dim")
parser.add_argument("-opt", "--optimizer",default='adam', help="optimizer algorithm (default adam)")
parser.add_argument("-lr", "--learning_rate",default=0.001, type=float, help="learning rate (default 0.001)")
parser.add_argument("--schedule_flag",default=False, type=bool, help="LR scheduler (default False)")
parser.add_argument("--dropout",default=0.1, type=float,help="dropout percentage (default 0.1)")

args = parser.parse_args()

                        
# info 
data_name= args.data
model_name=args.model_name
save_flag=args.save_model
print("save flag:", save_flag)

################
# model parameters
latent_dim=args.latent_dim
embedding_dim=args.embed_dim
BATCH_SIZE = args.batch_size

################
if data_name == 'wiki_tfidf' or data_name == 'wiki_sent' or data_name == 'bhatia_topics_tfidf':
    '''
    wiki data has 
            text with max length 30
            title with max length 8
    adding 2 to accommodate eostok and sostok
    '''
    max_text_len=30 + 2
    max_summary_len=8  + 2 



elif data_name == 'bhatia_topics':
        '''
        bhatia's topics has
                 topics with max length of 
                 labels with max length of 4
        adding 2 to accommodate for eostok and sostok
        '''
        max_text_len=12 + 2 
        max_summary_len=4 + 2 # for eos sos
else:
    print("check data_name")
    sys.exit(0)
   


# load data after clean, tokenization, padding and splitting (procssing done in file preprocess.py)
x_tr= np.load('data/'+data_name+'/x_tr.npy')
y_tr=np.load('data/'+data_name+'/y_tr.npy')

x_val=np.load('data/'+data_name+'/x_val.npy')
y_val=np.load('data/'+data_name+'/y_val.npy')



if args.sub_set:
    print("get only subset of the data")
    x_tr= x_tr[:args.sub_set]
    y_tr= y_tr[:args.sub_set]
    x_val = x_val[:int(args.sub_set*0.25)]
    y_val = y_val[:int(args.sub_set*0.25)]
   
x_tr, y_tr = shuffle(x_tr,y_tr, random_state=42)
x_val, y_val = shuffle(x_val,y_val, random_state=42)

print("number of training ", len(x_tr))
print("number of validating ", len(x_val))


# loading X tokenizer
with open('data/'+data_name+'/x_tokenizer.pickle', 'rb') as handle:
    x_tokenizer= pickle.load(handle)

# loading Y tokenizer
with open('data/'+data_name+'/y_tokenizer.pickle', 'rb') as handle:
    y_tokenizer= pickle.load(handle)
    
    

#size of vocabulary ( +1 for padding token)
x_voc   =  x_tokenizer.num_words + 1 #use num_words instead of word_index because I have large vocab and would like to keep the top-n only "check more"
vocab_inp_size = x_voc
#size of vocabulary
y_voc  =   y_tokenizer.num_words +1
vocab_tar_size = y_voc
print('number of vocab in x', x_voc)
print('number of vocab in y', y_voc)



BUFFER_SIZE = len(x_tr)

train_steps_per_epoch = len(x_tr)//BATCH_SIZE
val_steps_per_epoch = len(x_val)//BATCH_SIZE

print("x_tr shape", x_tr.shape)
print("y_tr shape", y_tr.shape)

dataset = tf.data.Dataset.from_tensor_slices((x_tr, y_tr)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

dataset_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).shuffle(BUFFER_SIZE)
dataset_val = dataset_val.batch(BATCH_SIZE, drop_remainder=True)


example_input_batch, example_target_batch = next(iter(dataset))
print('sample example_input_batch.shape:', example_input_batch.shape, ' sample example_target_batch.shape:', example_target_batch.shape)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = sup_methods.max_length(y_val), sup_methods.max_length(x_val)
print('max_length_targ', max_length_targ)
print('max_length_inp', max_length_inp)



if 'bigru_bahdanau_attention' == model_name:
    print("BiGRU + bahdanau attention") 
    
    encoder = model_archi.Encoder_bigru_attention(vocab_inp_size, embedding_dim, latent_dim, BATCH_SIZE, x_tokenizer )
    decoder = model_archi.Decoder_bigru_attention(vocab_tar_size, embedding_dim, latent_dim, BATCH_SIZE, y_tokenizer)

else:
    print("model name not found")
    sys.exit(0)
        

'''
Optimizer
'''
optimizer = get_optimizer(args.optimizer, args.learning_rate, args.latent_dim, args.schedule_flag)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


'''
Checkpoint to continue training
'''
checkpoint_dir = './training_checkpoints/'+ data_name
checkpoint_prefix_es = os.path.join(checkpoint_dir, model_name)

if args.checkpoint =='':
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
else:

    print("restore model from checkpoint")

    checkpoint_name = os.path.join(checkpoint_dir, args.checkpoint)
    print("path:", checkpoint_name)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    
    checkpoint.restore(checkpoint_name)
    # get reached epoch 
    new_epoch= get_epoch_num(args.checkpoint) + 1 # because get_epoch_num return the epoch that was reached and we want to start training from the next epoch    
    print("start training from epoch", new_epoch)
    
'''
Train ----------------
'''

@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    #print("inp shape", inp.shape, ' enc_hidden shape', enc_hidden.shape)
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([y_tokenizer.word_index['sostok']] * BATCH_SIZE, 1)

    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
      loss += loss_function(targ[:, t], predictions)
      #print("loss ", loss)
      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1) 

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss



print('TRAIN: number of batches per epoch:', len([batch for (batch, (inp, targ)) in enumerate(dataset.take(train_steps_per_epoch))]))
print('VAL: number of batches per epoch:', len([batch for (batch, (inp, targ)) in enumerate(dataset_val)]))


''' starting training with early stopping '''
best_loss= best_val_loss = 100
require_improvment = 5 # early stopping patience 
stop = False
losses=[]
val_losses=[]
rouge_scores=[]
if args.checkpoint != '':
    max_epochs = range(int(new_epoch), 1000)
else:
    max_epochs = range(1000)

for epoch in max_epochs:
  if stop == False:
      print('#'*50)
      print("EPOCH", epoch+1)
      print('#'*50)
      start = time.time()
    
      enc_hidden = encoder.initialize_hidden_state()
      total_loss = 0
      
      for (batch, (inp, targ)) in enumerate(dataset.take(train_steps_per_epoch)):

        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss
    
        if batch % 100 == 0:
          print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                       batch,
                                                       batch_loss.numpy()))
       
    
      
      avg_loss = total_loss / train_steps_per_epoch
      losses.append(np.float(avg_loss))
      print('Epoch {} Train Loss {:.4f}'.format(epoch + 1, avg_loss))
      
      print('Time taken  {:.1f} sec\n'.format(time.time() - start))
      
      ''' Validation '''

      # enumerate the validation data
      start = time.time()
      valid_total_loss = 0
      val_num_batch =0 
      print('val_steps_per_epoch', val_steps_per_epoch)
      #val_enc_hidden = encoder.initialize_hidden_state()
      val_enc_hidden = encoder.initialize_hidden_state()
      for (batch, (inp, targ)) in enumerate(dataset_val.take(val_steps_per_epoch)):
          
          batch_loss = validate(inp, targ, val_enc_hidden)
          valid_total_loss+= batch_loss
      
          val_num_batch = batch
    
        
      #show accumulative loss 
      avg_val_loss = valid_total_loss / val_steps_per_epoch
      val_losses.append(np.float(avg_val_loss))
      print('Epoch {} Valid Loss {:.4f}'.format(epoch + 1, avg_val_loss ))
      
      
      ## check if model improved or need to stop training
      if avg_val_loss < best_val_loss:
          best_val_loss = avg_val_loss
          best_loss= avg_loss
          best_epoch= epoch
          last_improvement= 0 
          
          # saving (checkpoint) 
          if args.save_model == True :
              file_path = checkpoint_prefix_es+ '_e_{}_valloss_{:.2f}_'.format(best_epoch, best_val_loss)
              checkpoint.save(file_prefix=file_path)
              print("save checkpoint:",file_path )
          else:
              print("no save")

          

      else:
          last_improvement+=1
      
      if last_improvement > require_improvment:
            print("No improvement found during the (self.require_improvement) last iterations, stopping.")
            print("loss:", losses)
            print("val_loss:",val_losses)
            # Break out from the loop.
            stop = True
      
  else:
      print('stopped training with early stopping Epoch {} Val_loss {:.2f}'.format(best_epoch, best_val_loss ))
      break
      
    
      

assert val_steps_per_epoch == val_num_batch + 1

