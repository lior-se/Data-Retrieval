import pandas as pd
from tqdm import tqdm
import time

# Load your DataFrame
file_path = "15000.xlsx"
df = pd.read_excel(file_path)
first_names=pd.read_excel("first-names.xlsx")
last_names=pd.read_excel("last-names.xlsx")
# Filter based on a specific condition
df_A = df[df.iloc[:, 3] == 'A'].iloc[:, 2]


from transformers import pipeline,AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-ner')
oracle = pipeline('ner', model='dicta-il/dictabert-ner', aggregation_strategy='simple')

# if we set aggregation_strategy to simple, we need to define a decoder for the tokenizer. Note that the last wordpiece of a group will still be emitted
from tokenizers.decoders import WordPiece
oracle.tokenizer.backend_tokenizer.decoder = WordPiece()

''' text one phrase
sentence = df_A[2]
#oracle(sentence)
start =time.time()
ner_output=oracle(sentence)
print(time.time()-start)
print(ner_output)
'''

def replace_per_entities(sentence, first_names, last_names):
    '''
    Detecte names and replace them with a first and a last name from the xlsx files
    '''
    ner_output = oracle(sentence)
    offset = 0
    for entity in ner_output:
        if entity['entity_group'] == 'PER':
            # Select random first and last name
            random_first_name = first_names.sample(n=1).iloc[0, 0]
            random_last_name = last_names.sample(n=1).iloc[0, 0]
            full_name = f"{random_first_name} {random_last_name}"

            # Calculate new start and end considering the current offset
            new_start = entity['start'] + offset
            new_end = entity['end'] + offset

            # Replace in the sentence
            sentence = sentence[:new_start] + full_name + sentence[new_end:]

            # Update offset based on the length difference after replacement
            offset += len(full_name) - (new_end - new_start)
    return sentence

# region Chunks splitting to fit maximum token size
def custom_convert_tokens_to_string(tokens, remove_unknown=True, unk_token="[UNK]"):
    """
    Manually convert a list of tokens to a string, handling special cases for subword tokenization schemes
    and optionally removing or replacing unknown tokens.

    Args:
    - tokens: List of tokens to convert.
    - remove_unknown: Boolean indicating whether to remove unknown tokens.
    - unk_token: The token used to represent unknown tokens.
    """
    # Optionally remove [UNK] tokens
    if remove_unknown:
        tokens = [token for token in tokens if token != unk_token]

    text = ' '.join(tokens)

    # Handling for BERT's WordPiece
    text = text.replace(' ##', '')

    # Handling for GPT-2's Byte Pair Encoding (BPE)
    text = text.replace('</w>', ' ').strip()

    # Handling for SentencePiece
    text = text.replace('▁', ' ').strip()

    # Normalize whitespace
    text = ' '.join(text.split())

    # Post-processing for punctuation and other special characters
    punctuations = [',', '.', '!', '?', '%', '$', '&', '*', '(', ')', '-', '=', '+', '/', '\\', ':', ';', '"', '\'']
    for p in punctuations:
        text = text.replace(f' {p}', p)

    return text


def split_text_into_token_chunks(text, max_tokens=510):
    """Split text into chunks that are within max_tokens length, considering special tokens."""
    initial_tokens = tokenizer.tokenize(text)

    # Check if chunking is necessary
    if len(initial_tokens) <= max_tokens:
        return [text]  # Return original text in a list if within limit

    chunks = []
    current_chunk = []
    current_length = 0
    for token in initial_tokens:
        if current_length + 1 > max_tokens:  # +1 for potential special token
            chunks.append(custom_convert_tokens_to_string(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(token)
        current_length += 1  # Assuming each token contributes at least 1 to length
    if current_chunk:  # Add the last chunk if any
        chunks.append(custom_convert_tokens_to_string(current_chunk))
    return chunks

# endregion

processed_texts = []
for text in tqdm(df_A[2:3]):
    chunks=split_text_into_token_chunks(text)
    processed_chunks = ""
    for chunk in chunks:
        processed_chunks += replace_per_entities(chunk, first_names, last_names)
    processed_texts.append(processed_chunks)


df_A_processed = pd.DataFrame(processed_texts)
output_file_path = "processed_df_A.xlsx"
df_A_processed.to_excel(output_file_path, index=False,header=False)

print("Processing complete and saved to:", output_file_path)
