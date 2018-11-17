from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def loadData():
    # Load context from the test data set and convert them into
    # the tf-idf vocabulary
    test_dataframe = pd.read_csv('../data/test.csv')
    context = test_dataframe['Context'].values.tolist()
    ground_truth = test_dataframe['Ground Truth Utterance'].values.tolist()
    distractors = []

    # Load distractors that will be used for Recall@k metrics
    for i in range(2, 11):
        distractors.append(test_dataframe[test_dataframe.columns[i]].values.tolist())

    return { 'context' : context, 'ground_truth': ground_truth, 'distractors': distractors }


def computeTfIdfMatrix(data):
    # Vectorize context
    context_vectorizer = TfidfVectorizer()
    context_vectorizer.fit(data['context'])
    return context_vectorizer

def computeMetric(min, max, k, context_vectorizer, data):
    context = data['context']
    ground_truth = data['ground_truth']
    distractors = data['distractors']
    correct_classifications = 0;
    score_vector = {}
    binary_classification = max <= 2
    dataset_size = len(context)

    for i in range(dataset_size):
        ctx_vector = context_vectorizer.transform([context[i]])
        gt_vector = context_vectorizer.transform([ground_truth[i]])
        # This vector will be used as base to compute r@k metrics
        gold_score = cosine_similarity(ctx_vector, gt_vector)
        score_vector['gt'] = gold_score.item(0)

        for e in range(max-1):
            dis_vector = context_vectorizer.transform([distractors[e][i]])
            score = cosine_similarity(dis_vector, ctx_vector)
            if binary_classification:
                # In the case of binary classification, just see if the
                # ground truth score is larger than the distractor one
                # and count it as a correct answer
                if (gold_score >= score):
                    correct_classifications = correct_classifications + 1
            else:
                # In case of classification over a larger set of responses
                # sort the responses by score and limit by k, if the ground
                # truth response is part of k, then count it as a correct answer
                score_vector['dt'+str(e)] = score.item(0)

        if not binary_classification:
            sorted_scores = sorted(score_vector.values(), reverse=True)
            sorted_scores = sorted_scores[:k]
            if score_vector['gt'] in sorted_scores:
                correct_classifications = correct_classifications + 1;

    recall = (correct_classifications * 100) / dataset_size
    return recall


def main():
    data = loadData()
    context_vectorizer = computeTfIdfMatrix(data)
    print 'Computing {} in {} R@{}: {}%'.format(1, 2, 1, computeMetric(1, 2, 1, context_vectorizer, data))
    print 'Computing {} in {} R@{}: {}%'.format(1, 10, 1, computeMetric(1, 10, 1, context_vectorizer, data))
    print 'Computing {} in {} R@{}: {}%'.format(1, 10, 5, computeMetric(1, 10, 5, context_vectorizer, data))

main()

