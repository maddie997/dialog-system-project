# cs230 final project

**tf-idf** Baseline model

**LSTM** First attempt to improve over the baseline

---

## How to run
1. Create a virtual env
`virtualenv env`
2. Activate env
`source env/bin/activate`
3. Install requirements
`pip install -r requirements.txt`
4. Get the data and embeddings and place them inside the **/data** directory:

 - Download training/validation/test sets from: https://drive.google.com/open?id=1SpwZUtk91GLCVPZYdZr9McFWvMnwjmPY

 - Download word embeddings: http://nlp.stanford.edu/data/glove.840B.300d.zip


## tf-idf
`python tfidf/model.py`

```
Computing 1 in 2 R@1: 75%

Computing 1 in 10 R@1: 50%

Computing 1 in 10 R@5: 77%
````
