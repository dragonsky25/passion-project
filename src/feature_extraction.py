from sklearn.feature_extraction.text import TfidfVectorizer


#converting tokens to a matrix of TF-IDF features
def extract_features(tokens, max_features=5000):
        vectorizer = TfidfVectorizer(max_features=max_features)
        vectors = vectorizer.fit_transform(tokens)
        return vectors
