## Emotion Prediction
#### This is a project aimed at recognizing emotions in English text using a predictive model. The emotions are classified into six groups: sadness, joy, love, anger, fear, and surprise. Logistic regression is used as a machine learning algorithm.
_____
**Overall, the code can be divided into 5 main steps:**
 1. Loading a dataset, making a few changes to it.
 2. Using lemmatizer and stemmer to collect tokens from the sentences. 
 3. Converting the tokens to a matrix of TF-IDF features.
 4. Splitting the matrix into train & test parts
 5. Using logistic regression to train the model on training data, evaluating the model
_____
**To run the code locally:**
 1. Clone the repository
 2. Install the dependencies using pip install -r requirements.txt
 3. Run the project through the main file