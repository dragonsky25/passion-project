import pickle

#load model from pickle file

model_path = "logistic_regression.pk1"

def load_model():
     with open(model_path,'rb') as file:
         model = pickle.load(file)

     return model