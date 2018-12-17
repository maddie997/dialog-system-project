
Updates from Madhulima:

The original codebase is from https://github.com/dennybritz/chatbot-retrieval.
This does not work with newer versions of TF (only TF 0.12.1). It has been
modified to enable the code to run with newer versions of tensorflow -- 1.11 and above.


Step 1: Do the following to prepare for running
Install code and download data per instructions
Virtualenv with Install Python 3.6.x, TF
pip3 install tensorflow==0.12.1 / pip3 install tensorflow-gpu==1.11 or a later version
(gpu highy recommended for this program)

Step 2: Download the dataset:
https://drive.google.com/open?id=0B_bZck-ksdkpVEtVc1R6Y01HMWM) and extract the acrhive into `./data`.

Step 3: Running Training:
change udc_hparams.py - set batch_size to 64, and eval_batch_size to 8 for running on a GPU with less than 6GB Memory

Note that multi-layer and bidirectional cant be both updated at the same time.


Change udc_train.py -   set the number of epochs to run for - set num_epochs to 100000 (will take 4-5 hours).
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
