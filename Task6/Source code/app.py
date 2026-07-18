import os
import pickle
from flask import Flask, request, render_template, redirect, url_for, session
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
# FIXED: Changed DUMMY_DATA to load_csv_data
from utils import clean_text, load_csv_data

app = Flask(__name__)
app.secret_key = "super_secret_flash_key_for_news_app"
MODEL_PATH = "fake_news_model.pkl"

def train_model():
    print("Training model on the real dataset... Please wait (5-10 seconds)...")
    # FIXED: Using our new CSV dataset loader function
    texts, labels = load_csv_data()
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b", min_df=2)),
        ("clf", MultinomialNB(alpha=1.0))
    ])
    pipeline.fit(texts, labels)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print("✨ Model training complete and saved successfully!")
    return pipeline

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f: 
                return pickle.load(f)
        except Exception: 
            pass
    return train_model()

model = load_model()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("news_text", "").strip()
        cleaned = clean_text(text)
        
        if cleaned:
            try:
                pred = model.predict([cleaned])[0]
                proba = model.predict_proba([cleaned])[0]
                confidence = round(max(proba) * 100, 2)
                result = "Real News" if pred == 1 else "Fake News"
            except Exception:
                result, confidence = "Uncertain (Insufficient Text Data)", 0.0
        else:
            result, confidence = "Invalid text input", 0.0
            
        session["result"] = result
        session["confidence"] = confidence
        session["text"] = text
        return redirect(url_for("index"))

    result = session.pop("result", None)
    confidence = session.pop("confidence", None)
    text = session.pop("text", None)
    
    return render_template("index.html", result=result, confidence=confidence, text=text)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}
    cleaned = clean_text(data.get("text", ""))
    if not cleaned: 
        return {"error": "empty or invalid text"}, 400
    try:
        pred = model.predict([cleaned])[0]
        proba = model.predict_proba([cleaned])[0]
        return {
            "label": "Real News" if pred == 1 else "Fake News",
            "confidence": round(float(max(proba)) * 100, 2)
        }
    except Exception:
        return {"label": "Unknown", "confidence": 0.0}

if __name__ == "__main__":
    app.run(debug=True, port=5000)
