from sklearn.feature_extraction.text import TfidfVectorizer


def extract_features(tokens):
        vectorizer = TfidfVectorizer()
def extract_features(tokens, max_features=None):
        vectorizer = TfidfVectorizer(max_features=max_features)
        vectors = vectorizer.fit_transform(tokens)
        return vectors
