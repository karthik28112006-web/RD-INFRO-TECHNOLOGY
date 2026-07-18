# utils.py

import io
import os
import pickle
import numpy as np
import pandas as pd
import requests
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import (
    MODEL_PATH,
    DATA_CACHE_PATH,
    DATA_URL,
    DATA_REQUEST_TIMEOUT,
    SAMPLE_SIZE_PER_CLASS,
    RANDOM_STATE,
)

# Curated neutral examples — public sentiment corpora rarely ship a clean
# "neutral" class, so this is combined with real fetched positive/negative data.
NEUTRAL_SEED_DATA = [
    "The package arrived on Tuesday.",
    "The product is grey and medium sized.",
    "I received the item as described in the listing.",
    "The meeting is scheduled for 3 PM on Thursday.",
    "This is a standard model with basic features.",
    "The store opens at 9 AM and closes at 6 PM.",
    "The report contains twelve pages and three appendices.",
    "The device weighs approximately two pounds.",
    "The train departs from platform four at noon.",
    "This document was last updated in March.",
    "The recipe requires flour, water, and salt.",
    "The conference room is located on the second floor.",
    "The battery lasts approximately ten hours on a full charge.",
    "The invoice was sent to the billing department yesterday.",
    "The software is compatible with Windows and Mac operating systems.",
    "The event begins at noon and concludes at five.",
    "The box includes a manual, two cables, and a charger.",
    "The file format supports most common editors.",
    "The company was founded in 2005 and is headquartered downtown.",
    "The survey was distributed to two hundred participants.",
    "The building has four elevators and two stairwells.",
    "The shipment is expected to arrive within five business days.",
    "The thermostat displays the current temperature in Celsius.",
    "The library is open from Monday through Saturday.",
    "The spreadsheet has six columns and forty rows.",
    "The car has a five-speed manual transmission.",
    "The website was migrated to a new server last week.",
    "The contract is valid for a period of twelve months.",
    "The printer uses standard A4 paper.",
    "The flight has a layover in Chicago.",
    "The museum exhibit features artifacts from the 1800s.",
    "The parking garage has three levels.",
    "The password must contain at least eight characters.",
    "The bridge spans roughly two kilometers.",
    "The warehouse stores inventory for three regional branches.",
    "The lecture covered chapters four through six.",
    "The subscription renews automatically every month.",
    "The form requires a signature on the last page.",
    "The garden has both vegetable and flower sections.",
    "The update log lists seven minor changes.",
    "The app was released on the third of this month.",
    "The setting can be changed from the preferences menu.",
    "The dashboard displays usage statistics for the past week.",
    "This version requires at least 4GB of available storage.",
    "The tool integrates with three third-party platforms.",
]

# Domain-specific seeds to bridge the gap between informal tweet-style
# training data and formal product/app/software review language.
PRODUCT_POSITIVE_SEED = [
    "Highly recommend this tool, saved me hours of manual work.",
    "This new update is incredible, everything works perfectly.",
    "The interface is smooth and very intuitive to use.",
    "Excellent app, exactly what our team needed.",
    "This feature works flawlessly and saves a lot of time.",
    "Great tool for tracking, very easy to set up.",
    "The new design is clean and works perfectly.",
    "This update fixed everything, runs smoothly now.",
    "Fantastic tool, made our workflow so much easier.",
    "The performance improvements are impressive, everything feels faster.",
    "Really impressed with how reliable this software is.",
    "This app exceeded my expectations, works great every time.",
    "Setup was quick and the tool performs beautifully.",
    "The new version is a huge improvement, highly satisfied.",
    "Everything works as expected, very smooth experience overall.",
    "This solution streamlined our entire process, love it.",
    "Outstanding update, the app feels much more responsive now.",
    "The tool integrates seamlessly and saves significant time.",
    "Very pleased with the results, this tool works perfectly.",
    "This app made social media tracking effortless and efficient.",
]

PRODUCT_NEGATIVE_SEED = [
    "This app keeps crashing every time I open it.",
    "The update broke several features that used to work fine.",
    "Extremely buggy software, unusable in its current state.",
    "The interface is confusing and slows down my workflow.",
    "This tool is unreliable and wastes more time than it saves.",
    "The new version is slower and full of glitches.",
    "Terrible update, nothing works as it should anymore.",
    "The app freezes constantly and loses unsaved work.",
    "Very disappointed, this tool doesn't do what it promises.",
    "The software is riddled with bugs and poor performance.",
    "This update made everything worse, very frustrating experience.",
    "The tool failed to sync and caused data loss.",
    "Constant errors make this app nearly impossible to use.",
    "The performance is sluggish and the interface is clunky.",
    "This feature simply does not work as advertised.",
    "The app crashed repeatedly during basic tasks.",
    "Poorly designed update that broke core functionality.",
    "This tool is a waste of time, riddled with issues.",
    "The software lags badly and often becomes unresponsive.",
    "Very frustrating, this update caused more problems than it solved.",
]

# Minimal offline fallback if the remote dataset cannot be reached or parsed
_FALLBACK_DATA = [
    ("I love this product, it works great!", "positive"),
    ("Absolutely fantastic experience, highly recommend.", "positive"),
    ("Best purchase I've made all year.", "positive"),
    ("Amazing service and friendly staff.", "positive"),
    ("I hate this, it's a complete waste of money.", "negative"),
    ("Terrible experience, would not recommend.", "negative"),
    ("Worst customer service I've ever had.", "negative"),
    ("This broke after one day, very disappointed.", "negative"),
] + [(t, "neutral") for t in NEUTRAL_SEED_DATA]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map known source column names to a standard text/label schema."""
    cols = {c.lower().strip(): c for c in df.columns}

    text_col = next((cols[c] for c in ("twitts", "text", "tweet", "review") if c in cols), None)
    label_col = next((cols[c] for c in ("sentiment", "label", "target") if c in cols), None)

    if text_col is None or label_col is None:
        raise ValueError("Unrecognized dataset schema: missing text/label columns")

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]
    df["label"] = df["label"].map(_coerce_label)
    df = df.dropna(subset=["text", "label"])

    if df.empty or "label" not in df.columns:
        raise ValueError("Dataset normalization produced no usable rows")

    return df.reset_index(drop=True)


def _coerce_label(raw):
    """Normalize varied label encodings (0/1, 'pos'/'neg', etc.) to positive/negative."""
    if isinstance(raw, str):
        val = raw.strip().lower()
        if val in ("1", "pos", "positive", "4"):
            return "positive"
        if val in ("0", "neg", "negative"):
            return "negative"
        return None
    try:
        num = float(raw)
        return "positive" if num > 0 else "negative"
    except (TypeError, ValueError):
        return None


def _download_remote_dataset() -> pd.DataFrame:
    response = requests.get(DATA_URL, timeout=DATA_REQUEST_TIMEOUT)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    return _normalize_columns(df)


def _load_dataset() -> tuple[pd.DataFrame, str]:
    """Load dataset from local cache if present and valid, else download and cache it."""
    if os.path.exists(DATA_CACHE_PATH):
        try:
            df = pd.read_csv(DATA_CACHE_PATH)
            if {"text", "label"}.issubset(df.columns) and len(df) > 0:
                return df, "cache"
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            pass

    df = _download_remote_dataset()
    df.to_csv(DATA_CACHE_PATH, index=False)
    return df, "remote"


def _sample_balanced(df: pd.DataFrame, per_class: int, random_state: int) -> pd.DataFrame:
    """Sample up to `per_class` rows per label without relying on groupby.apply,
    which can drop the grouping column across different pandas versions."""
    frames = []
    for label_value in df["label"].dropna().unique():
        subset = df[df["label"] == label_value]
        n = min(len(subset), per_class)
        frames.append(subset.sample(n=n, random_state=random_state))
    return pd.concat(frames, ignore_index=True)


def _build_training_data() -> tuple[list, list, dict]:
    """Combine fetched real-world data with curated neutral + domain-specific
    product-review seeds. Class imbalance is left to `class_weight="balanced"`
    in the classifier rather than duplicating rows, which previously caused
    overfitting on a handful of repeated neutral sentences."""
    try:
        df, source = _load_dataset()

        if "label" not in df.columns or "text" not in df.columns:
            raise ValueError("Loaded dataset missing required columns")

        balanced_src = _sample_balanced(df, SAMPLE_SIZE_PER_CLASS, RANDOM_STATE)

        texts = balanced_src["text"].astype(str).tolist()
        labels = balanced_src["label"].astype(str).tolist()

        meta_source = source

    except (requests.RequestException, ValueError, KeyError,
            pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as err:
        texts, labels = [list(t) for t in zip(*_FALLBACK_DATA)]
        meta_source = "offline_fallback"

    # Always add curated neutral + domain-specific seeds regardless of source,
    # to bridge the gap between informal tweet data and formal review language.
    texts += NEUTRAL_SEED_DATA
    labels += ["neutral"] * len(NEUTRAL_SEED_DATA)

    texts += PRODUCT_POSITIVE_SEED
    labels += ["positive"] * len(PRODUCT_POSITIVE_SEED)

    texts += PRODUCT_NEGATIVE_SEED
    labels += ["negative"] * len(PRODUCT_NEGATIVE_SEED)

    meta = {
        "source": meta_source,
        "n_samples": len(texts),
        "live_data": meta_source != "offline_fallback",
    }
    return texts, labels, meta


def _train_and_save_model(model_path: str) -> dict:
    texts, labels, meta = _build_training_data()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
        )),
    ])

    pipeline.fit(texts, labels)

    bundle = {"pipeline": pipeline, "meta": meta}
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    return bundle


def load_or_train_model(model_path: str = MODEL_PATH, force_retrain: bool = False) -> dict:
    """Load a persisted model bundle if available, otherwise fetch data and train."""
    if force_retrain:
        for path in (model_path, DATA_CACHE_PATH):
            if os.path.exists(path):
                os.remove(path)
        return _train_and_save_model(model_path)

    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)
            if isinstance(bundle, dict) and "pipeline" in bundle:
                return bundle
        except (pickle.UnpicklingError, EOFError, AttributeError):
            pass

    return _train_and_save_model(model_path)


def predict_sentiment(pipeline: Pipeline, text: str) -> dict:
    """Predict sentiment label and per-class confidence scores for the given text."""
    cleaned = text.strip()
    if not cleaned:
        return {"label": "neutral", "confidence": 0.0, "probabilities": {}}

    proba = pipeline.predict_proba([cleaned])[0]
    classes = pipeline.classes_
    prob_map = {cls: float(p) for cls, p in zip(classes, proba)}

    predicted_label = classes[int(np.argmax(proba))]
    confidence = prob_map[predicted_label]

    return {
        "label": predicted_label,
        "confidence": confidence,
        "probabilities": prob_map,
    }