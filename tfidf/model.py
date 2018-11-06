from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load context from the test data set and convert them into
# the tf-idf vocabulary

test_data = pd.read_csv('../data/test.csv')

print test_data.shape

