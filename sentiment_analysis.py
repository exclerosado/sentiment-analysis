import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from train_data import train

# removing stopwords from the database
def remove_stopwords(data):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    spllited = data.split()
    result = [word for word in spllited if word.lower() not in stopwords]

    return ' '.join(result)


# apllying stem on database
def apply_stem(data):
    stemmer = nltk.RSLPStemmer()
    result = [stemmer.stem(word) for word in data.split()]

    return ' '.join(result)


def clean_data():
    # first cleaning proccess - removing stopwords
    no_stopwords = []
    for index in train:
        no_stopwords.append(tuple((remove_stopwords(index[0]), index[1])))

    # last cleaning proccess - stemming
    with_stem = []
    for idx in no_stopwords:
        with_stem.append(tuple((apply_stem(idx[0]), idx[1])))

    return with_stem


def extract_features(data):
    words = set(data.split())

    return {word: True for word in words}


def train_classifier(cleaned_data):
    feature_set = [(extract_features(text), sentiment) for (text, sentiment) in cleaned_data]
    classifier = NaiveBayesClassifier.train(feature_set)
    
    return classifier


def analyze_sentiment(classifier, text):
    cleaned_text = apply_stem(remove_stopwords(text))
    features = extract_features(cleaned_text)

    return classifier.classify(features)


X_train = clean_data()
classifier = train_classifier(X_train)

X_test = str(input('Enter a sentence: '))
sentiment = analyze_sentiment(classifier, X_test)
print(sentiment)

print(f'Accuracy: {round(accuracy(classifier, [(extract_features(text), sentiment) for (text, sentiment) in X_train]), 2) * 100}%')