from flask import Flask
from flasgger import Swagger
from flask_restful import Api, Resource
import pandas as pd
import joblib
import spacy
import en_core_web_sm
import cleaning as cl


app = Flask(__name__)
api = Api(app)

template = {
  "swagger": "2.0",
  "info": {
    "title": "Tags generator of the stackoverflow website"
  }
}

swagger = Swagger(app, template=template)
# Load pre-trained models
model_path = "models/"
vectorizer = joblib.load(model_path + "tfidf_vectorizer.pkl", 'r')
multilabel_binarizer = joblib.load(model_path + "multilabel_binarizer.pkl", 'r')
model = joblib.load(model_path + "logit_nlp_model.pkl", 'r')

class Autotag(Resource):
    def get(self, question):
       
        # Clean the question sent
        
        nlp = spacy.load('en_core_web_md')
        pos_list = ["NOUN","PROPN"]
        rawtext = question
        cleaned_question = cl.text_cleaner(rawtext, nlp, pos_list, "english")
        
        # Apply saved trained TfidfVectorizer
        X_tfidf = vectorizer.transform([cleaned_question])
        
        # Perform prediction
        predict = model.predict(X_tfidf)
        # Inverse multilabel binarizer
        tags_predict = multilabel_binarizer.inverse_transform(predict)
        
        
            
        # Results
        results = {}
        results['Predicted_Tags'] = tags_predict
        
        return results, 200


api.add_resource(Autotag, '/autotag/<question>')

if __name__ == "__main__":
	app.run(debug=True)