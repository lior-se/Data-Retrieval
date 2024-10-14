def extract_words(input_file, output_file):
    """
    Extracts words from a .conllu file based on specified POS tags and writes them to a .txt file.
    Each segment in the input file separated by a line starting with '#' is written to a new line in the output file.

    :param input_file: Path to the input .conllu file
    :param output_file: Path to the output .txt file
    """
    pos_tags = {'NOUN', 'VERB', 'PROPN', 'ADJ'}

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        words = []

        for line in infile:
            # Check for a new segment
            if line.startswith('# text ='):
                if words:
                    outfile.write(' '.join(words) + '\n')  # Write the current segment's words
                    words = []  # Start a new list for the next segment
                continue

            # Skip lines that don't contain data
            if line.strip() == '':
                continue

            columns = line.split('\t')
            # Check if the line has enough columns and if the POS tag matches
            if len(columns) > 3 and columns[3] in pos_tags:
                words.append(columns[2])  # Append the word from the third column

        # Write the last segment's words if any
        if words:
            outfile.write(' '.join(words) + '\n')


# Example usage
extract_words('A_lemma.conllu', 'A_lemma.txt')
extract_words('B_lemma.conllu', 'B_lemma.txt')
extract_words('C_lemma.conllu', 'C_lemma.txt')
#extract_words('C_text_OWSout_10.conllu', 'C_lemma10.txt')