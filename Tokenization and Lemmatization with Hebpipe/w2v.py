import numpy as np
import pandas as pd
from tqdm import tqdm


def read_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip().split() for line in file]


def write_w2v_matrix(document_list, file_name, words, word_vectors, group):
    word_index_map = {word: idx for idx, word in enumerate(words)}  # For efficient lookups
    document_indices = group  # Create the index list here
    w2v_data = []

    for doc in tqdm(document_list):
        document_vector = np.zeros_like(word_vectors[0])

        for word in doc:
            if word in word_index_map:
                index = word_index_map[word]
                document_vector += word_vectors[index]

        w2v_data.append(document_vector)

    w2v = pd.DataFrame(w2v_data, index=group, columns=[f'{i}' for i in range(len(word_vectors[0]))])
    w2v.to_excel(f'{file_name}_w2v.xlsx', encoding='utf-8-sig', index=True)



A_text = read_document("A_text.txt")
B_text = read_document("B_text.txt")
C_text = read_document("C_text.txt")
A_lemma = read_document("A_lemma.txt")
B_lemma = read_document("B_lemma.txt")
C_lemma = read_document("C_lemma.txt")

A_text_SW = read_document("A_text_with_SW.txt")
B_text_SW = read_document("B_text_with_SW.txt")
C_text_SW = read_document("C_text_with_SW.txt")
A_lemma_SW = read_document("save/A_lemma.txt")
B_lemma_SW = read_document("save/B_lemma.txt")
C_lemma_SW = read_document("save/C_lemma.txt")

with open("wiki-w2v-pos/words_list.txt", 'r', encoding='utf-8') as w2v_file:
    words = [line.strip() for line in w2v_file]
word_vectors = np.load("wiki-w2v-pos/words_vectors.npy")

file_path = "C:/Users/Lior/Downloads/15000.xlsx"
df = pd.read_excel(file_path)
# drop empty cells
document_ids = df.dropna(subset=[df.columns[1]]).iloc[:, 1].tolist()
A_IDs=document_ids[:5000]
B_IDs=document_ids[5000:10000]
C_IDs=document_ids[10000:]

write_w2v_matrix(A_text, "A_text", words, word_vectors,A_IDs)
write_w2v_matrix(B_text, "B_text", words, word_vectors,B_IDs)
write_w2v_matrix(C_text, "C_text", words, word_vectors,C_IDs)
write_w2v_matrix(A_lemma, "A_lemma", words, word_vectors,A_IDs)
write_w2v_matrix(B_lemma, "B_lemma", words, word_vectors,B_IDs)
write_w2v_matrix(C_lemma, "C_lemma", words, word_vectors,C_IDs)

write_w2v_matrix(A_text_SW, "A_text_with_SW", words, word_vectors,A_IDs)
write_w2v_matrix(B_text_SW, "B_text_with_SW", words, word_vectors,B_IDs)
write_w2v_matrix(C_text_SW, "C_text_with_SW", words, word_vectors,C_IDs)
write_w2v_matrix(A_lemma_SW, "A_lemma_with_SW", words, word_vectors,A_IDs)
write_w2v_matrix(B_lemma_SW, "B_lemma_with_SW", words, word_vectors,B_IDs)
write_w2v_matrix(C_lemma_SW, "C_lemma_with_SW", words, word_vectors,C_IDs)
