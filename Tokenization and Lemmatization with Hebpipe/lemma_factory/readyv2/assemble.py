def assemble_files_ordered(output_file):
    """
    Assembles files named from 'A_text_OWSout_1.conllu' to 'A_text_OWSout_25.conllu'
    into a single file with UTF-8 encoding.

    :param output_file: Name of the output file where contents will be assembled
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i in range(1, 26):  # Loop from 1 to 25
            filename = f'A_text_OWSout/A_text_OWSout_{i}.conllu'  # Construct filename for each iteration
            try:
                with open(filename, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read() + '\n')
            except FileNotFoundError:
                print(f"File not found: {filename}")
                continue

# Call the function with the output file name
assemble_files_ordered('A_lemma.conllu')
