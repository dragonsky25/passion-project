import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import load_data
from word_filter_func import word_filter
from feature_extraction import extract_features


def main():
  #collecting tokens
  filepath = '/Passion project/text.csv'
  data = load_data(filepath)
  cleaned_texts = [word_filter(sentence) for sentence in data["text"]]
  cleaned_texts_joined = [' '.join(tokens) for tokens in cleaned_texts]
  feature_matrix = extract_features(cleaned_texts_joined)
  training, testing, = train_test_split(
  feature_matrix, test_size = 0.2, shuffle = False
  )
  return testing

if __name__ == "__main__":
  print(main())
