# utils.py
import pandas as pd
import requests
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

DATA_URL = "https://raw.githubusercontent.com/rashida048/Some-NLP-Projects/master/movie_dataset.csv"
REQUIRED_COLS = ["index", "title", "genres", "vote_average"]

def load_data():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(DATA_URL, headers=headers, timeout=10)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))

    df.columns = [c.strip().lower() for c in df.columns]
    if "index" not in df.columns:
        df = df.reset_index().rename(columns={"index": "index"})

    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = "" if col in ("title", "genres") else 0

    df = df[REQUIRED_COLS]
    df["title"] = df["title"].astype(str).str.strip()
    df["genres"] = df["genres"].astype(str).str.strip().str.replace("|", " ", regex=False)
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["title", "genres"])
    df = df[(df["title"] != "") & (df["genres"] != "")]
    df = df.reset_index(drop=True)
    return df

def precompute_similarity(df):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["genres"])
    return cosine_similarity(matrix, matrix)

def get_recommendations(title, df, sim_matrix, top_n=5):
    if title not in df["title"].values:
        return []
    idx = df[df["title"] == title].index[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:top_n]
    return df["title"].iloc[[i[0] for i in scores]].tolist()

def get_genre_recommendations(chosen_genre, df, top_n=5):
    pattern = r"\b" + re.escape(chosen_genre) + r"\b"
    mask = df["genres"].str.contains(pattern, case=False, regex=True, na=False)
    filtered = df[mask].sort_values("vote_average", ascending=False)
    return filtered["title"].head(top_n).tolist()