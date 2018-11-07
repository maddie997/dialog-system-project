from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load context from the test data set and convert them into
# the tf-idf vocabulary

test_dataframe = pd.read_csv('../data/test.csv')
context = test_dataframe['Context'].values.tolist()

context_vectorizer = TfidfVectorizer()
# tokenize and build vocab
context_vectorizer.fit(context)

for c in context:
    context_vector = context_vectorizer.transform([c])
    # this vector should be used to run cosine distance
    # against responses and collect data to determine recall@1-9
