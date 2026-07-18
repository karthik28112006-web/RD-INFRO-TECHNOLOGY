import re
import string
import csv
import os

CSV_PATH = "real_news_dataset.csv"

def clean_text(text):
    if not text: 
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def load_csv_data():
    """Reads real data directly from the manually extracted CSV file safely."""
    texts, labels = [], []
    if not os.path.exists(CSV_PATH):
        # Emergency fallback if your file is named incorrectly in the folder
        return [("Government announces policy", 1), ("Aliens control world", 0)]
        
    with open(CSV_PATH, "r", encoding="utf-8", errors="ignore") as f:
        # Handles long paragraphs and internal quotation marks safely
        reader = csv.reader(f, delimiter=',', quotechar='"')
        try:
            next(reader)  # Skip the CSV header row
        except StopIteration:
            return texts, labels

        for row in reader:
            # Layout format check: row[0]=ID, row[1]=title, row[2]=text, row[3]=label
            if len(row) >= 4:
                text_content = row[2].strip()
                label_str = row[3].strip().upper()
                
                if text_content and label_str in ["REAL", "FAKE"]:
                    texts.append(clean_text(text_content))
                    labels.append(1 if label_str == "REAL" else 0)
    return texts, labels

