# Přiřazení sentimentu pomocí VADERu (s ChatGPT :) )

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

#stažení VADER lexikonu
nltk.download('vader_lexicon')

#Inicializace analyzátoru
sia = SentimentIntensityAnalyzer()

#načtení dat
df = pd.read_csv("Data/FRIENDS_SCRIPT_CLEAN.csv", encoding="utf-8")


# Výpočet sentimentu pro každý text
# vytvoří čtyři nové sloupce: neg, neu, pos, compound
sentiment_scores = df["text"].apply(lambda x: sia.polarity_scores(str(x)))

#Přidání těchto hodnot do tabulky
df["neg"] = sentiment_scores.apply(lambda x: x["neg"])
df["neu"] = sentiment_scores.apply(lambda x: x["neu"])
df["pos"] = sentiment_scores.apply(lambda x: x["pos"])
df["compound"] = sentiment_scores.apply(lambda x: x["compound"])

#Přidání kategorie podle skóre
def classify_sentiment(score):
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"

df["sentiment"] = df["compound"].apply(classify_sentiment)

#Uložení nového datasetu
df.to_csv("Data/FRIENDS_SCRIPT_SENTIMENT.csv", index=False)

print("Hotovo! Nový soubor uložen jako FRIENDS_SCRIPT_SENTIMENT.csv")