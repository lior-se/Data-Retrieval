import pandas as pd
from tqdm import tqdm
from googletrans import Translator
import csv


file_path = "15000.xlsx"
df = pd.read_excel(file_path)


df_A = df[df.iloc[:, 3] == 'A'].iloc[:, 2]
df_B = df[df.iloc[:, 3] == 'B'].iloc[:, 2]
df_C = df[df.iloc[:, 3] == 'C'].iloc[:, 2]


translator = Translator()

def split_text(text, max_length=3000):
    """Split text into chunks that are shorter than max_length, splitting at the last possible newline."""
    if len(text) <= max_length:
        return [text]
    chunks = []
    while text:
        split_at = text.rfind('\n', 0, max_length) + 1  # Find last newline before max_length
        if split_at == 0:  # No newline found; split at max_length
            split_at = text.rfind(' ', 0, max_length) + 1  # Try to split at space if no newline
            if split_at == 0:  # No space found; force split at max_length
                split_at = max_length
        chunk, text = text[:split_at], text[split_at:]
        chunks.append(chunk)
    return chunks

def translate_text(text, dest_language='en'):
    """Translate text, handling long texts by splitting and rejoining them."""
    chunks = split_text(text)
    translated_chunks = [translator.translate(chunk, dest=dest_language).text for chunk in chunks]
    return ' '.join(translated_chunks)

def translate_dataframe_column(df, dest_language='en'):
    """Translate each text in a DataFrame column, handling long texts appropriately."""
    translated_texts = [translate_text(text, dest_language) for text in tqdm(df, desc="Translating")]
    return pd.Series(translated_texts)


translated_df_A = translate_dataframe_column(df_A)
translated_df_A.to_csv('A_translated.csv',index=False,header=False)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk


translated_df_A = pd.read_csv('A_translated.csv', header=None, squeeze=True)

nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()

def get_neg_pos_scores(text):
    scores = sid.polarity_scores(text)
    return scores['neg'], scores['pos'],scores['neu']

neg_scores, pos_scores,neu_scores = zip(*translated_df_A.apply(get_neg_pos_scores))

average_neg = sum(neg_scores) / len(neg_scores)
average_pos = sum(pos_scores) / len(pos_scores)
average_neu = sum(neu_scores) / len(neu_scores)

print(f"Average Negative Score: {average_neg:.5f}")
print(f"Average Positive Score: {average_pos:.5f}")
print(f"Average Neutral Score: {average_neu:.5f}")