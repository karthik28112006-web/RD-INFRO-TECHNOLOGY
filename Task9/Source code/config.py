# config.py

# Streamlit page configuration
PAGE_CONFIG = {
    "page_title": "AI Sentiment Analysis",
    "page_icon": "🧠",
    "layout": "centered",
    "initial_sidebar_state": "collapsed",
}

# UI theme constraints
PRIMARY_COLOR = "#6C63FF"
BACKGROUND_COLOR = "#0E1117"
TEXT_COLOR = "#FAFAFA"
FONT = "sans serif"

APP_TITLE = "🧠 AI Sentiment Analysis"
APP_SUBTITLE = "Analyze the sentiment of any text using a Logistic Regression model."
INPUT_PLACEHOLDER = "Type or paste your text here..."
MAX_CHARS = 2000

# Model / persistence paths
MODEL_PATH = "model.pkl"
DATA_CACHE_PATH = "data_cache.csv"

# Remote dataset (public, real-world tweets labeled positive/negative)
DATA_URL = "https://raw.githubusercontent.com/laxmimerit/twitter-data/master/twitter4000.csv"
DATA_REQUEST_TIMEOUT = 10  # seconds

# Training constants
SAMPLE_SIZE_PER_CLASS = 300  # cap per class for speed and balance
RANDOM_STATE = 42

CONFIDENCE_THRESHOLD = 0.6

SENTIMENT_LABELS = {
    "positive": "😊 Positive",
    "negative": "😞 Negative",
    "neutral": "😐 Neutral",
}

SENTIMENT_COLORS = {
    "positive": "#2ECC71",
    "negative": "#E74C3C",
    "neutral": "#F1C40F",
}