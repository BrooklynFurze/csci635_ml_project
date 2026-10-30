# this is the example download for the dataset I haven't set up a Kaggle account yet
# So this is a work in progress
# kagglehub is only needed to re-download/re-sample the raw dataset (see below);
# the sampled csv is already in data/, so it's not imported on every run
# import kagglehub
# from kagglehub import KaggleDatasetAdapter
import pandas as pd
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from evaluation import compare_methods
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer,ENGLISH_STOP_WORDS,CountVectorizer
# # Download latest version
# path = Path(kagglehub.dataset_download("andrewmvd/steam-reviews"))

# print(path)

#----------------------------------------
# # Sample the original dataset
# inputPath = path / "dataset.csv"
# outputPath = Path(__file__).resolve().parent.parent / "data" / "steam_reviews_sampled.csv"

# df = pd.read_csv(inputPath)
# print(df.head())

# n = sum(1 for line in open(inputPath,encoding="utf-8")) - 1
# sampleSize = 50000 # 50000 for now, adjust later if needed
# skip = sorted(random.sample(range(1,n+1),n-sampleSize))
# df = pd.read_csv(inputPath,skiprows=skip,encoding="utf-8")
# df.to_csv(outputPath,index=False)
#------------------------------------------

df = pd.DataFrame()
totalStopwords = set(ENGLISH_STOP_WORDS)
# for sentiment analysis keep the negative words
negativeStopwords = set(w for w in totalStopwords if "n't" in w or "no" in w or "not" in w)
finalStopwords = list(totalStopwords - negativeStopwords)


def preprocessing():
    sampled_data_path = Path(__file__).resolve().parent.parent / "data" / "steam_reviews_sampled.csv"
    df = pd.read_csv(sampled_data_path)
    # clean the dataset
    df = df.drop(columns=['app_id']).drop_duplicates().dropna()

    print(df.info())
    print(df.isnull().sum())
    print(df["review_score"].value_counts(dropna=False))
    print(df["review_text"].head(10))
    
    # split the train and test sets
    X = df["review_text"] # not sure if we should include app names and review votes
    y = df["review_score"]
    
    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,)
    
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    
    return X_train, X_test, y_train, y_test

    

def bag_of_words(X_train, X_test):
    vectorizer = CountVectorizer(stop_words=finalStopwords,lowercase=True)
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    return X_train_bow,X_test_bow

def tf_idf(X_train,X_test):
    vectorizer = TfidfVectorizer(stop_words=finalStopwords,lowercase=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf,X_test_tfidf

# rename once we figure out what the third word analysis type we're doing is
def third_parsing_version(): 
    print("Placeholder")

def main():
    X_train,X_test,y_train,y_test = preprocessing()

    # same classifier for every method, so score differences come from the text representation only.
    # class_weight="balanced" because only ~18% of reviews are negative: unweighted, the model rarely
    # predicts "negative" at the default 0.5 cutoff and the threshold-based metrics (macro F1,
    # negative recall) end up measuring that cutoff rather than how good the representation is
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    # (name, vectorizer function, classifier, train texts, test texts) -- the vectorizer functions are
    # passed in (not their output) because cross-validation has to re-fit the vocabulary/idf on each
    # training fold. the majority baseline always predicts "positive": the floor every method has to beat
    # TODO: add the lemmatized variants here, e.g. ("tf-idf + lemmatization", tf_idf, model, x_train_lem, x_test_lem)
    methods = [
        ("majority baseline", bag_of_words, DummyClassifier(strategy="most_frequent"), X_train, X_test),
        ("bag of words", bag_of_words, model, X_train, X_test),
        ("tf-idf", tf_idf, model, X_train, X_test),
    ]
    compare_methods(methods, y_train, y_test)


if __name__ == '__main__':
    main()