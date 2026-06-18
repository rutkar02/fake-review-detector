from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings,logging

# 1. Block standard Python warnings matching the Hugging Face message
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

def load_data():
    return load_dataset(
        "theArijitDas/Fake-Reviews-Dataset"
        )

def prepare_data():
    dataset = load_data()    
    
    df = dataset["train"].to_pandas()

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]
    return X,y,vectorizer,df

X,y,vectorizer,df = prepare_data()

def train_model(X,y):
    X_train,X_test,y_train,y_test= train_test_split( # data is splitted here
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    model = LogisticRegression(
        max_iter=1000
    )
    model.fit(X_train,y_train) # model is trained here
    return model

model = train_model(X,y)

def predict_review(review):
    review_vector = vectorizer.transform([review]) # it tranforms our text into suitable features in order for model to process # we used [] because vectorizer expects a list of documents thats what we trained it on
    prediction = model.predict(review_vector)[0] # we use [0] coz answer is returned as an array
    probability = model.predict_proba(review_vector)[0]
    
    confidence = max(probability)
    return prediction,confidence