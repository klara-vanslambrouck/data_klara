# --- SENTIMENT A EMOCE PRO SERIÁL FRIENDS --- (ChatGPT)

#SENTIMENT

import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch

# Inicializace progress baru
tqdm.pandas()

# Načtení dat
df = pd.read_csv("Data/FRIENDS_SCRIPT_CLEAN.csv")

# Načtení modelu
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=0 if torch.cuda.is_available() else -1
)

# Funkce pro vyhodnocení sentimentu (omezíme délku textu)
def get_sentiment(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return None
    result = sentiment_model(text[:512])[0]  # Roberta neumí víc než cca 512 tokenů
    return result["label"]

# Výpočet sentimentu s progress barem
df["sentiment"] = df["text"].progress_apply(get_sentiment)

# Uložení výsledků
df.to_csv("Data/FRIENDS_SENTIMENT_EMOTIONS.csv", index=False)

print("Hotovo. Výsledky jsou uložené v souboru FRIENDS_SENTIMENT.csv")

#EMOCE

import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Inicializace progress baru
tqdm.pandas()

# Načtení dat se sentimentem
df = pd.read_csv("Data/FRIENDS_SENTIMENT_EMOTIONS.csv")

# Inicializace modelu pro emoce
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=False
)

# Aplikace modelu na každý text s progress barem
df["emotion"] = df["text"].progress_apply(lambda x: emotion_model(x[:512])[0]["label"] if isinstance(x, str) else None)

# Uložení výsledků zpět do stejného souboru
df.to_csv("Data/FRIENDS_SENTIMENT_EMOTIONS.csv", index=False)

print("Hotovo! Emoce byly doplněny do FRIENDS_SENTIMENT_EMOTIONS.csv.")
