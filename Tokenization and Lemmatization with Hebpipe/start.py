import pandas as pd
import re
import subprocess
import csv

file_path = "C:/Users/Lior/Downloads/15000.xlsx"

df = pd.read_excel(file_path)

with open('stopswords_list_extend.txt', 'r', encoding='utf-8') as file:
    stopwords = [line.strip() for line in file]


def clean_text(text):
    if isinstance(text, str):

        hebrew_text = re.sub(r'(?<![\u0590-\u05FF])"|"(?![\u0590-\u05FF])', '', text)  # remove " not between hebrew
        hebrew_text = re.sub(r'[^\u0590-\u05FF\"\s]', '', hebrew_text)

        words = hebrew_text.split()

        # remove stopwords
        words_without_stopwords = [word for word in words if word not in stopwords]

        return ' '.join(words_without_stopwords)
    else:
        return text

def clean_text_no_stopwords(text):
    if isinstance(text, str):

        hebrew_text = re.sub(r'(?<![\u0590-\u05FF])"|"(?![\u0590-\u05FF])', '', text)  # remove " not between hebrew
        hebrew_text = re.sub(r'[^\u0590-\u05FF\"\s]', '', hebrew_text)
        hebrew_text=hebrew_text.split()

        return ' '.join(hebrew_text)
    else:
        return text

# second column
df.iloc[:, 2] = df.iloc[:, 2].apply(clean_text_no_stopwords)

df_A = df[df.iloc[:, 3] == 'A']
df_B = df[df.iloc[:, 3] == 'B']
df_C = df[df.iloc[:, 3] == 'C']


df_A.iloc[:, 2].to_csv('A_text_with_SW.txt', index=False, header=False, quoting=csv.QUOTE_NONE)
df_B.iloc[:, 2].to_csv('B_text_with_SW.txt', index=False, header=False, quoting=csv.QUOTE_NONE)
df_C.iloc[:, 2].to_csv('C_text_with_SW.txt', index=False, header=False, quoting=csv.QUOTE_NONE)


#subprocess.run(['python', '-m', 'hebpipe', '-wpl', '*.txt'])

#df.to_excel('cleaned_file.xlsx', index=False)

