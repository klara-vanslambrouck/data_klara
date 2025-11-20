# --- SENTIMENT A EMOCE PRO SERIÁL FRIENDS --- Chat GPT

import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch

tqdm.pandas()  # aktivace progress baru


# ============================
# 1) SENTIMENT
# ============================

# Načtení dat
df = pd.read_csv("Data/FRIENDS_SCRIPT_CLEAN.csv")

# Pipeline pro sentiment
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=0 if torch.cuda.is_available() else -1,
    max_length=512,   
    truncation=True        
)

def get_sentiment(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return None
    try:
        result = sentiment_model(text)[0] 
        return result["label"]
    except Exception:
        return None

df["sentiment"] = df["text"].progress_apply(get_sentiment)

# Uložení mezivýsledku
df.to_csv("Data/FRIENDS_SENTIMENT_EMOTIONS.csv", index=False)
print("Sentiment hotový → uložen do FRIENDS_SENTIMENT_EMOTIONS.csv")



# ============================
# 2) EMOCE
# ============================

# Načtení mezivýsledku
df = pd.read_csv("Data/FRIENDS_SENTIMENT_EMOTIONS.csv")

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    device=0 if torch.cuda.is_available() else -1,
    return_all_scores=False,
    max_length=512,
    truncation=True
)

def get_emotion(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return None
    try:
        result = emotion_model(text)[0]
        return result["label"]
    except Exception:
        return None

df["emotion"] = df["text"].progress_apply(get_emotion)

# Uložení finálního výstupu
df.to_csv("Data/FRIENDS_SENTIMENT_EMOTIONS.csv", index=False)
print("Emoce hotové → vše uloženo do FRIENDS_SENTIMENT_EMOTIONS.csv")