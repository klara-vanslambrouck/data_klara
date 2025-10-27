import spacy
import re
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv("friends_script.csv")
df.head()
