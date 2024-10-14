def count_unique_words(filename):
    with open(filename, 'r',encoding='utf-8') as file:
        content = file.read()
        words = content.split()
        unique_words = set(words)
        return len(unique_words)


unique_word_count = count_unique_words('A_lemma.txt')
print(f"Number of unique words: {unique_word_count}")
