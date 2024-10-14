import sys

from scipy.sparse import lil_matrix
import math
import pandas as pd
from scipy.sparse import csr_matrix

class BM25:
    """
    Best Match 25.

    Parameters
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    Attributes
    ----------
    tf_ : list[dict[str, int]]
        Term Frequency per document. So [{'hi': 1}] means
        the first document contains the term 'hi' 1 time.

    df_ : dict[str, int]
        Document Frequency per term. i.e. Number of documents in the
        corpus that contains the term.

    idf_ : dict[str, float]
        Inverse Document Frequency per term.

    doc_len_ : list[int]
        Number of terms per document. So [3] means the first
        document contains 3 terms.

    corpus_ : list[list[str]]
        The input corpus.

    corpus_size_ : int
        Number of documents in the corpus.

    avg_doc_len_ : float
        Average number of terms for documents in the corpus.
    """

    def __init__(self, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1

    def fit(self, corpus):
        """
        Fit the various statistics that are required to calculate BM25 ranking
        score using the corpus given.

        Parameters
        ----------
        corpus : list[list[str]]
            Each element in the list represents a document, and each document
            is a list of the terms.

        Returns
        -------
        self
        """
        tf = []
        df = {}
        idf = {}
        doc_len = []
        corpus_size = 0
        for document in corpus:
            corpus_size += 1
            doc_len.append(len(document))

            # compute tf (term frequency) per document
            frequencies = {}
            for term in document:
                term_count = frequencies.get(term, 0) + 1
                frequencies[term] = term_count

            tf.append(frequencies)

            # compute df (document frequency) per term
            for term, _ in frequencies.items():
                df_count = df.get(term, 0) + 1
                df[term] = df_count

        for term, freq in df.items():
            #idf[term] = math.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))
            idf[term] = math.log((corpus_size + 1) / df[term])
        self.tf_ = tf
        self.df_ = df
        self.idf_ = idf
        self.doc_len_ = doc_len
        self.corpus_ = corpus
        self.corpus_size_ = corpus_size
        self.avg_doc_len_ = sum(doc_len) / corpus_size
        return self

    def search(self, query):
        scores = [self._score(query, index) for index in range(self.corpus_size_)]
        return scores

    def _score(self, query, index):
        score = 0.0

        doc_len = self.doc_len_[index]
        frequencies = self.tf_[index]
        for term in query:
            if term not in frequencies:
                continue

            freq = frequencies[term]
            numerator = self.idf_[term] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len_)
            score += (numerator / denominator)

        return score

    def create_tf_idf_matrix(self):
        """
        Create a TF-IDF matrix using BM25 scores.

        Returns
        -------
        matrix : scipy.sparse.lil_matrix
            A matrix of size (number of documents) x (number of unique terms)
            where each element represents the BM25 score of the term in the document.
        """
        # number of unique terms
        terms = list(self.df_.keys())
        term_indices = {term: index for index, term in enumerate(terms)}

        # sparse matrix.
        matrix = lil_matrix((self.corpus_size_, len(terms)))

        for doc_index, document in enumerate(self.corpus_):
            for term in document:
                if term in term_indices:
                    term_index = term_indices[term]
                    matrix[doc_index, term_index] = self._score([term], doc_index)

        return matrix, terms


def read_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip().split() for line in file]


def create_matrix(corpus):

    # build a word count dictionary, so we can remove words that appear only once
    word_count_dict = {}
    for text in corpus:
        for token in text:
            word_count_dict[token] = word_count_dict.get(token, 0) + 1

    sorted_word_counts = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)
    with open("word_frequencies.txt", "w", encoding="utf-8") as file:
        for term, count in sorted_word_counts:
            file.write(f"{term}: {count}\n")

    texts = [[token for token in text if word_count_dict[token] > 45] for text in corpus]
    bm25 = BM25()
    bm25.fit(texts)
    return bm25.create_tf_idf_matrix()

'''
A_Lemma=create_matrix("A_lemma.txt")
print(A_Lemma.shape)
B_Lemma=create_matrix("B_lemma.txt")
C_Lemma=create_matrix("C_lemma.txt")
A=create_matrix("A_text_OWS.txt")
print(A.shape)
B=create_matrix("B_text_OWS.txt")
C=create_matrix("C_text_OWS.txt")
'''



corpus = read_document("A_text.txt")
corpus.extend(read_document("B_text.txt"))
corpus.extend(read_document("C_text.txt"))
matrix, terms=create_matrix(corpus)
print("matrix created")
print(matrix.shape)

A_matrix = matrix[:5000, :]   # First 5000 rows
B_matrix = matrix[5000:10000, :]  # Next 5000 rows
C_matrix = matrix[10000:15000, :]  # Last 5000 rows


#region Excel Lemma


file_path = "C:/Users/Lior/Downloads/15000.xlsx"
df = pd.read_excel(file_path)
# drop empty cells
document_ids = df.dropna(subset=[df.columns[1]]).iloc[:, 1].tolist()

lemmot= pd.DataFrame(matrix.toarray(), columns=terms,index=document_ids)
letters = ['A'] * 5000 + ['B'] * 5000 + ['C'] * (len(lemmot) - 10000)
lemmot['Group'] = letters
print(lemmot.shape)
print("before A")
lemmot.iloc[:5000].to_excel('A_text.xlsx', index=True,encoding='utf-8')
print("A_lemma excel created")
lemmot.iloc[5000:10000].to_excel('B_text.xlsx', index=True,encoding='utf-8')
lemmot.iloc[10000:].to_excel('C_text.xlsx', index=True,encoding='utf-8')
print(lemmot.iloc[:5, :5])
print(lemmot.shape)

#endregion

'''vector possible creation

document_vectors = []

for i, vector in enumerate(matrix):
    # Determine the letter based on the index
    if i < 5000:
        letter = 'A'
    elif i < 10000:
        letter = 'B'
    else:
        letter = 'C'

    # Append the tuple to the list
    document_vector = (document_ids[i], vector, letter)
    document_vectors.append(document_vector)

A_lemma = document_vectors[:5000]
print(A_lemma[0][0])
print(A_lemma[0][1])
print(A_lemma[0][2])
print(A_lemma[4999][0])
print(A_lemma[4999][1])
print(A_lemma[4999][2])
'''




#region InfoGain--------------------------------------------------------------------------------------------



from scipy.stats import entropy
import numpy as np



from scipy.sparse import issparse

def calculate_entropy(labels):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=2)

def info_gain(X, y):
    if issparse(X):
        X = X.toarray()

    total_entropy = calculate_entropy(y)
    n_rows = float(X.shape[0])
    ig_scores = []

    for col in range(X.shape[1]):
        unique_values, counts = np.unique(X[:, col], return_counts=True)
        weighted_entropy = 0.0

        for value, count in zip(unique_values, counts):
            subset_prob = count / n_rows
            subset_entropy = calculate_entropy(y[X[:, col] == value])
            weighted_entropy += subset_prob * subset_entropy

        ig = total_entropy - weighted_entropy
        ig_scores.append(ig)

    return ig_scores



def calculate_IG(X,Y):
    X_dense = X.toarray() if isinstance(X, csr_matrix) else X
    scores = info_gain(X_dense,Y)
    ig_scores_dict = dict(zip(terms, scores))
    return ig_scores_dict


'''
#whole matrix categories 1 2 3 5000 each
Y = np.array([1]*5000 + [2]*5000 + [3]*5000)
ig_scores=calculate_IG(matrix,Y)
with open("IG_scores.txt", "w", encoding="utf-8") as file:
    for term, score in ig_scores.items():
        file.write(f"{term}: {score}\n")
'''

Y_lemmots = np.arange(1, 5001)
ig_scores_A=calculate_IG(A_matrix,Y_lemmots)
with open("IG_scores_Atext.txt", "w", encoding="utf-8") as file:
    for term, score in ig_scores_A.items():
        file.write(f"{term}: {score}\n")

ig_scores_B=calculate_IG(B_matrix,Y_lemmots)
with open("IG_scores_Btext.txt", "w", encoding="utf-8") as file:
    for term, score in ig_scores_B.items():
        file.write(f"{term}: {score}\n")
ig_scores_C=calculate_IG(C_matrix,Y_lemmots)
with open("IG_scores_Ctext.txt", "w", encoding="utf-8") as file:
    for term, score in ig_scores_C.items():
        file.write(f"{term}: {score}\n")
#endregion

#region mutual info----------------------------------------------------------------------------------

from sklearn.feature_selection import mutual_info_classif

def calculate_MIG(X,Y):
    X_dense = X.toarray() if isinstance(X, csr_matrix) else X
    scores = mutual_info_classif(X_dense, Y, discrete_features=True)
    mig_scores_dict = dict(zip(terms, scores))
    return mig_scores_dict

Y_lemmots = np.arange(1, 5001)


mig_scores_A=calculate_MIG(A_matrix,Y_lemmots)
with open("MIG_scores_Atext.txt", "w", encoding="utf-8") as file:
    for term, score in mig_scores_A.items():
        file.write(f"{term}: {score}\n")

mig_scores_B=calculate_MIG(B_matrix,Y_lemmots)
with open("MIG_scores_Btext.txt", "w", encoding="utf-8") as file:
    for term, score in mig_scores_B.items():
        file.write(f"{term}: {score}\n")
mig_scores_C=calculate_MIG(C_matrix,Y_lemmots)
with open("MIG_scores_Ctext.txt", "w", encoding="utf-8") as file:
    for term, score in mig_scores_C.items():
        file.write(f"{term}: {score}\n")

#endregion -------------------------------------------------------------------------------------------

#region IG MIG
data = {
    'Term': list(ig_scores_A.keys()),
    'IG_Score': list(ig_scores_A.values()),
    'MIG_Score': list(mig_scores_A.values())
}

# Create DataFrame
df = pd.DataFrame(data)

# Write DataFrame to xlsx file
df.to_excel("IG_MIG_Atext.xlsx", index=False)

data_B = {
        'Term': list(ig_scores_B.keys()),
        'IG_Score': list(ig_scores_B.values()),
        'MIG_Score': list(mig_scores_B.values())
}

    # Create DataFrame for B set
df_B = pd.DataFrame(data_B)

# Write DataFrame to xlsx file for B set
df_B.to_excel("IG_MIG_Btext.xlsx", index=False)

data_C = {
        'Term': list(ig_scores_C.keys()),
        'IG_Score': list(ig_scores_C.values()),
        'MIG_Score': list(mig_scores_C.values())
}

# Create DataFrame for C set
df_C = pd.DataFrame(data_C)

# Write DataFrame to xlsx file for C set
df_C.to_excel("IG_MIG_Ctext.xlsx", index=False)

#endregion

#region SHAP-----------------------------------------------------------------------------------------

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import shap
import xgboost
def calculate_shap(X,Y):
    X_dense = X.toarray().astype('float')

    bst = xgboost.train(
        {"learning_rate": 0.01, "max_depth": 4}, xgboost.DMatrix(X_dense, label=Y), 1000
    )
    shap_values = bst.predict(xgboost.DMatrix(X_dense), pred_contribs=True)
    print("values")

    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
    mean_shap = dict(zip(terms, mean_abs_shap_values))
    return mean_shap

# shap_matrix=calculate_shap(matrix,Y)
#with open("mean_abs_SHAP.txt", "w", encoding="utf-8") as file:
#    for feature_name, value in zip(terms, mean_abs_shap_values):
#        file.write(f"{feature_name}: {value}\n")


shap_matrix_A=calculate_shap(A_matrix,Y_lemmots)
shap_matrix_B=calculate_shap(B_matrix,Y_lemmots)
shap_matrix_C=calculate_shap(C_matrix,Y_lemmots)
#endregion SHAP

#region final excel scores
data = {
    'Term': list(ig_scores_A.keys()),
    'IG_Score': list(ig_scores_A.values()),
    'MIG_Score': list(mig_scores_A.values()),
    'SHAP': list(shap_matrix_A.values())
}

# Create DataFrame
df = pd.DataFrame(data)

# Write DataFrame to xlsx file
df.to_excel("IG_MIG_SHAP_Atext.xlsx", index=False)

data_B = {
        'Term': list(ig_scores_B.keys()),
        'IG_Score': list(ig_scores_B.values()),
        'MIG_Score': list(mig_scores_B.values()),
        'SHAP': list(shap_matrix_B.values())
}

    # Create DataFrame for B set
df_B = pd.DataFrame(data_B)

# Write DataFrame to xlsx file for B set
df_B.to_excel("IG_MIG_SHAP_Btext.xlsx", index=False)

data_C = {
        'Term': list(ig_scores_C.keys()),
        'IG_Score': list(ig_scores_C.values()),
        'MIG_Score': list(mig_scores_C.values()),
        'SHAP': list(shap_matrix_C.values())
}

# Create DataFrame for C set
df_C = pd.DataFrame(data_C)

# Write DataFrame to xlsx file for C set
df_C.to_excel("IG_MIG_SHAP_Ctext.xlsx", index=False)

#endregion

'''
corpus = read_document("A_text.txt")
corpus.extend(read_document("B_text.txt"))
corpus.extend(read_document("C_text.txt"))
matrix, terms=create_matrix(corpus)

print(A.iloc[:5, :5])
print(A.shape)
'''