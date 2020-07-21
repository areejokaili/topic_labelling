

# Automatic Generation of Topic labels 

This repository contains the source code and data used for the paper:<br>
<br>
**Automatic Generation of Topic Labels** (2020) Areej Alokaili, Nikolaos Aletras and Mark Stevenson in *Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval(SIGIR ’20)*, July 25–30, 2020, Virtual Event, China. https://doi.org/10.1145/3397271.3401185  [Pre-print](https://arxiv.org/abs/2006.00127)
<br>
<br>

## (A) Install required libraries

Python 3.6.9 is used.
  1. TensorFlow V2
  1. NumPy
  1. scikit-learn
  1. ipykernel

***below libraries needed for evaluation only. You can skip if you want to do different evaluation metric other than BERTScore*** 
  1. sentencepiece
  1. transformers
  1. bert-score 
  1. matplotlib
  1. pandas
  
   
use `pip install -r requirements.txt` to install all needed libraries


## (B) Training
To run the model (data are processed and ready, only training is needed):
- Navigate to topic_labelling/
   1. To train the model with [**inputs**=top-30 terms from wikipedia article and **outputs**=wikipedia titles]
   ```
   python train_tf.py -m 'bigru_bahdanau_attention'  -d 'wiki_tfidf'
   ``` 
   2. To train the model with [**inputs**=first-30 words from wikipedia article and **outputs**=wikipedia titles] (refer to paper for details).
   ```
   python train_tf.py -m 'bigru_bahdanau_attention'  -d 'wiki_sent'
   ``` 

- Training will stop if no improvment is recorded and all checkpoints will be saved in **training_checkpoint/data_name/** .

* #### Training options are detailed in the code or run 
```
python train_tf.py -h
```

## (C) Inference (generate titles/labels)

1. Generate TITLES for a subset of **wikipedia** articles (1000 articles)
 ```
 python test_tf.py -m 'bigru_bahdanau_attention' -s 1000 -d 'wiki_tfidf' --load 'NAME_OF_CHECKPOINT'
 ``` 
*replace **NAME_OF_CHECKPOINT** with the name of your checkpoint. For example, python test_tf.py -d 'wiki_tfidf' -m 'bigru_bahdanau_attention' --load bigru_bahdanau_attention_e_1_valloss_2.19_-2 

2. Generate LABELS for **bhatia_topics**
`python test_tf.py -m 'bigru_bahdanau_attention' -s 1000 -d 'wiki_tfidf' --load 'NAME_OF_CHECKPOINT' -te 'bhatia_topics'`
	
3. Generate LABELS for **bhatia_topics_tfidf**
`python test_tf.py -m 'bigru_bahdanau_attention' -s 1000 -d 'wiki_tfidf' --load 'NAME_OF_CHECKPOINT' -te 'bhatia_topics_tfidf'`




4. Predictions, golds, and topics will be stored at **results/data_name/** as
	- [<em>model_name</em>]_pred.out
	- [<em>model_name</em>]_gold.out
	- [<em>model_name</em>]_topics.out.
## (D) Evaluation 
1. To measure the similarity between predicted and gold labels, 
`python compute_bertscore.py -g results/path_to_gold_file.out -p results/path_to_predict_file.out` 
2. Output includes precision (P), recall (R) and f-score (F).

## **Repository hierarchy**  

* **train_tf.py** code to train the labelling network.
* **test_tf.py** code to generate new titles/labels.
* **model_archi_tf.py** neural network structure defind here.
* **support_methods.py** contain some method needed methods through out the system.
* **extract_additional_terms_for_topics.ipynb** notebook showing the steps taken to filter topic/labels pairs based on the overall human rating and matching them to similar documents to extract additional terms for bhatia_topics_tfidf.
* **compute_bertscore.py**: script to compute pairwise BERTScore between predicted titles/labels and gold titles/labels.  
* **data**

  1.	**wiki_tfidf**: contain files after preprocessing that are ready to be passed to the model. 
  1.  **wiki_sent**: contain files after preprocessing that are ready to be passed to the model. 
  1.  **bhatia_topics**: contains a csv file with two columns (column1: topic labels, columns2: topic's top 10 terms).
  1.	**bhatia_topics_tfidf**: contains a csv file with two columns (column1: topic labels, columns2: topic's top 10 terms +20 terms from similar document (the 20 terms are extract using file **extract_additional_topic_terms.ipynb**).
	
* **results**: this is where the model's output are saved in text files.
*	**training_checkpoints**: model checkpoints are saved here.
