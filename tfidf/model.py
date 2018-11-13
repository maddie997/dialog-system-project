from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load context from the test data set and convert them into
# the tf-idf vocabulary

# TODO: will need to clean that up, remove stop words and __eou__ / __eot__
test_dataframe = pd.read_csv('../data/test.csv')
context = test_dataframe['Context'].values.tolist()
ground_truth = test_dataframe['Ground Truth Utterance'].values.tolist()
distractor = test_dataframe['Distractor_0'].values.tolist()

# Vectorize context, ground truth, and distractor
context_vectorizer = TfidfVectorizer()
context_vectorizer.fit(context)

for i in range(len(context)):
    context_vector = context_vectorizer.transform([context[i]])
    ground_truth_vector = context_vectorizer.transform([ground_truth[i]])
    distractor_vector = context_vectorizer.transform([distractor[i]])
    print cosine_similarity(context_vector, ground_truth_vector)
    print cosine_similarity(context_vector, distractor_vector)
    break
