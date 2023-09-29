# Import necessary libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load your dataset
data = pd.read_csv('Book1.csv')

# Preprocessing
data['query'] = data['query'].str.lower()

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(data['query'])
y_train = data['intent']

# Choose a classifier
classifier = MultinomialNB()
# Train the classifier
classifier.fit(X_train_tfidf, y_train)

# Create a FastAPI app
app = FastAPI()

# Define a request model
class InputText(BaseModel):
    text: str

# Define an endpoint for training the model (POST request)
@app.post("/train_model/")
async def train_model(input_text: InputText):
    try:
        # Add the new data to the dataset and retrain the model
        new_data = pd.DataFrame({'query': [input_text.text.lower()], 'intent': ["NewIntent"]})
        data = pd.concat([data, new_data], ignore_index=True)
        data['query'] = data['query'].str.lower()
        X_train_tfidf = tfidf_vectorizer.fit_transform(data['query'])
        y_train = data['intent']
        classifier.fit(X_train_tfidf, y_train)
        return {"message": "Model trained successfully with new data."}
    except Exception as e:
        return {"error": str(e)}

# Define an endpoint for making intent predictions (GET request)
@app.get("/predict_intent/")
async def predict_intent(text: str, threshold: float = 0.7):
    try:
        input_text = text.lower()
        input_text = tfidf_vectorizer.transform([input_text])
        probabilities = classifier.predict_proba(input_text)[0]
        
        max_prob = max(probabilities)
        if max_prob < threshold:
            return {"intent": "I did not understand"}
            
        intent = classifier.classes_[np.argmax(probabilities)]
        return {"intent": intent}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
