# this is the example download for the dataset I haven't set up a Kaggle account yet
# So this is a work in progress
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer,ENGLISH_STOP_WORDS,CountVectorizer
# # Download latest version
path = Path(kagglehub.dataset_download("andrewmvd/steam-reviews"))

print(path)

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
    df = pd.read_csv("../data/steam_reviews_sampled.csv")
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

# Lemmatization converts words into their base form (e.g. running and ran both become run)
# This will test to see if TF IDF or Bag Of Words affectiveness are impacted by lemmatization
def lemmatization(X_train, x_test): 
    print("Placeholder")

# models trained on four variants: tf idf w/ and w/o lemmatization, bag of words w/ and w/o lemmatization
def main():
    X_train,X_test,y_train,y_test = preprocessing()
    X_train_tfidf,X_test_tfidf = tf_idf(X_train,X_test)
    X_train_bow,X_test_bow = bag_of_words(X_train,X_test)
    
    # TODO: train the model for each vectorization method and compare the results
    

if __name__ == '__main__':
    main()