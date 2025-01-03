from sklearn.feature_extraction.text import TfidfVectorizer


def extract_features(tokens):
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(tokens)
        return vectors
