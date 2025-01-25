import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import load_data
from word_filter_func import word_filter
from feature_extraction import extract_features
from logistic_regression import log_regression
from sklearn.metrics import accuracy_score, classification_report


def main():
  #collecting tokens
  #data loading
  filepath = '/Passion project/text.csv'
  data = load_data(filepath)

  #collecting tokens
  cleaned_texts = [word_filter(sentence) for sentence in data["text"]]
  cleaned_texts_joined = [' '.join(tokens) for tokens in cleaned_texts]

  #extracting features
  feature_matrix = extract_features(cleaned_texts_joined)
  training, testing, = train_test_split(
  feature_matrix, test_size = 0.2, shuffle = False
  #splitting matrix into train & test
  labels = data['emotion']

  #splitting matrix into train & test
  x_train, x_test, y_train, y_test = train_test_split(
  feature_matrix, labels ,test_size = 0.2, shuffle = False
      feature_matrix, labels ,test_size = 0.2, shuffle = False
  )
  return testing

  #predicting labels
  y_predict = log_regression(x_train, y_train, x_test)
  return y_predict

  for text, predicted, actual in zip(data['text'][:10], y_predict[:10], y_test[:10]):
    print(f"Text: {text}.\nPredicted emotion: {predicted}.\nActual emotion: {actual}.\n{'-' * 50}")

  print(accuracy_score(y_test, y_predict))
  print(classification_report(y_test, y_predict))
  #print(accuracy_score(y_test, y_predict))
  #print(classification_report(y_test, y_predict))

  return y_predict, y_test

if __name__ == "__main__":
  print(main())
  y_pred, y_test = main()
  print("Predicted labels:", y_pred)
  print("Actual labels:", y_test)
  #print("Predicted labels:", y_pred)
  #print("Actual labels:", y_test)
