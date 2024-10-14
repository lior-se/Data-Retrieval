import pandas as pd

file_path = "top_3000_most_freq_wiki.csv"

df = pd.read_csv(file_path)
stopwords = set(df.iloc[:, 1])

def remove_stopwords_from_file(file_path, stopwords):
    with open(file_path, 'r',encoding='utf-8') as file:
        lines = file.readlines()

    # Remove stopwords
    cleaned_lines = []
    for line in lines:
        words = line.split()
        cleaned_words = [word for word in words if word not in stopwords]
        cleaned_lines.append(" ".join(cleaned_words))
    with open(file_path, 'w',encoding='utf-8') as file:
        file.write("\n".join(cleaned_lines))


remove_stopwords_from_file("A_text.txt",stopwords)
remove_stopwords_from_file("B_text.txt",stopwords)
remove_stopwords_from_file("C_text.txt",stopwords)
