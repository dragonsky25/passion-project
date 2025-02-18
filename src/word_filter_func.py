import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer


#processing text data and splitting it into tokens using NLP techniques
def word_filter(text:str) -> list[str]:
  tokens = word_tokenize(text)
  stemmer = PorterStemmer()
  lemmatizer = WordNetLemmatizer()
  stop_words = set(stopwords.words('english'))
  filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
  stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
  final_tokens = [lemmatizer.lemmatize(word) for word in stemmed_words]
  return final_tokens