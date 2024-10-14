import pandas as pd
from tqdm import tqdm
tqdm.pandas()


file_path = "15000.xlsx"
df = pd.read_excel(file_path)

df_A = df[df.iloc[:, 3] == 'A'].iloc[:, 2]
df_B = df[df.iloc[:, 3] == 'B'].iloc[:, 2]
df_C = df[df.iloc[:, 3] == 'C'].iloc[:, 2]




#region Hebert initialization
from transformers import AutoTokenizer, AutoModel, pipeline
tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT_sentiment_analysis") #same as 'avichr/heBERT' tokenizer
model = AutoModel.from_pretrained("avichr/heBERT_sentiment_analysis")

# how to use?
sentiment_analysis = pipeline(
    "sentiment-analysis",
    model="avichr/heBERT_sentiment_analysis",
    tokenizer="avichr/heBERT_sentiment_analysis",
    return_all_scores = True
)
#endregion


def split_text_into_token_chunks(text, max_tokens=510):  # Leave some space for special tokens
    """Split text into chunks that are up to max_tokens tokens long."""
    tokens = tokenizer.tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for token in tokens:
        if current_length + len(tokenizer.tokenize(token)) + 1 > max_tokens:
            chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(token)
        current_length += len(tokenizer.tokenize(token))
    if current_chunk:
        chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
    return chunks

import time

def get_neg_pos_scores(text):
    chunks = split_text_into_token_chunks(text)
    all_scores = {'neutral': [], 'positive': [], 'negative': []}

    for chunk in chunks:
        results = sentiment_analysis(chunk)
        for result in results[0]:  # Iterate through each sentiment result for the chunk
            # Extract and accumulate scores for each sentiment
            all_scores[result['label']].append(result['score'])

    # Calculate average scores for each sentiment by averaging the accumulated scores
    avg_scores = {
        label: sum(scores) / len(scores) if scores else 0
        for label, scores in all_scores.items()
    }

    return avg_scores['negative'], avg_scores['positive'], avg_scores['neutral']


neg_scores, pos_scores,neu_scores = zip(*df_C.progress_apply(get_neg_pos_scores))
average_neg = sum(neg_scores) / len(neg_scores)
average_pos = sum(pos_scores) / len(pos_scores)
average_neu = sum(neu_scores) / len(neu_scores)

print(f"Average Negative Score: {average_neg:.5f}")
print(f"Average Positive Score: {average_pos:.5f}")
print(f"Average Neutral Score: {average_neu:.5f}")

