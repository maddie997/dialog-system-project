Updates from Madhulima:

Requirements -- Python 3.6 and Tensorflow 1.12

The original codebase is from https://github.com/dennybritz/chatbot-retrieval.
This does not work with newer versions of TF (only TF 0.12.1). It has been
modified to enable the code to run with newer versions of tensorflow -- 1.11 and above.
See ../chatbot-retrieval/README.md

Step 0: Follow the instructions in the chatbot-retrieval folder to install the
       right Python/tensorflow environment and downlaod the dataset.

Step 1: Generate the BERT updated dataset from the CSV files
     1.1 Install the  bert serving server and client (see https://pypi.org/project/bert-serving-server/)
     	 pip install bert-serving-server  # server
	 pip install bert-serving-client  # client
     1.2 test bert serving server --> run "python3 bert_test.py" embeddings of the sentences passed should be printed.
     1.3 mv data to scripts directory
     1.4 python prepare_data.py       (run Python 3.6) --> Note that this will take about 7 hours. The tfrecords file
     	 for test and validation files generated will be about 500MG, and the training file will be over 6GB. 
	 
Step 2. Set values in udc_hparams.py
     tf.flags.DEFINE_integer("max_context_len", 768)
     tf.flags.DEFINE_integer("max_utterance_len", 768)
     This just defines the size of the bert embeddings. In the
     dataset itself, the context/responses are truncated to 160 before
     conversion to BERT embeddings.

Step 3: Running Training:
     change udc_hparams.py - set batch_size to 64, and eval_batch_size to 8 for running on a GPU with less than 6GB Memory
     Change udc_train.py -   set the number of epochs to run for - set num_epochs to 100000.
python3 udc_train.py
	To save run logs: "python3 udc_train.py 2>&1 | tee -a lstm_512_log.txt"
	Note the subdirectory, e.g., runs/1542404725,  where the trained model is checkpointed. 

Step 4: Running Test:
     python3 udc_test.py --model_dir=./runs/1542404725



-------
Old README below (From original code at https://github.com/dennybritz/chatbot-retrieval)


## Retrieval-Based Conversational Model in Tensorflow (Ubuntu Dialog Corpus)

#### [Please read the blog post for this code](http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow)

#### Overview

The code here implements the Dual LSTM Encoder model from [The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems](http://arxiv.org/abs/1506.08909).

#### Setup

This code uses Python 3 and Tensorflow >= 0.9. Clone the repository and install all required packages:

```
pip install -U pip
pip install numpy scikit-learn pandas jupyter
```

#### Get the Data


Download the train/dev/test data [here](https://drive.google.com/open?id=0B_bZck-ksdkpVEtVc1R6Y01HMWM) and extract the acrhive into `./data`.


#### Training

```
python udc_train.py
```


#### Evaluation

```
python udc_test.py --model_dir=...
```


#### Evaluation

```
python udc_predict.py --model_dir=...
```
