import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class SMSClassifier:
  def __init__(self, model_path, vector_path):
    self.model = self.load_model(model_path)
    self.vector = self.load_vector(vector_path)
    self.lemmatizer = WordNetLemmatizer()
  
  def load_model(self, model_path):
    with open(model_path, "rb") as f:
      model = pickle.load(f)
    return model
  
  def load_vector(self, vector_path):
    with open(vector_path, "rb") as f:
      vector = pickle.load(f)
    return vector
  
  def preprocess_sms(self, sms):
    cleaned_sms = re.sub("[^a-zA-Z]", " ", sms)
    lower_sms = cleaned_sms.lower()
    token_sms = lower_sms.split()
    token_sms = [word for word in token_sms if word not in stopwords.words("english")]
    token_sms = [self.lemmatizer.lemmatize(word) for word in token_sms]
    token_sms = " ".join(token_sms)
    
    return token_sms
  
  def predict(self, sms):
    preprocessed_sms = self.preprocess_sms(sms)
    sms_list = [preprocessed_sms]
    sms_vector = self.vector.transform(sms_list)
    prediction = self.model.predict(sms_vector)
    return prediction.item()

if __name__ == "__main__":
  model_path = "bin/model.pkl"
  vector_path = "bin/vector.pkl"

  classifier = SMSClassifier(model_path, vector_path)
  input_sms = "I want to go to you house at 2 PM"
  result = classifier.predict(input_sms)
  print(result)