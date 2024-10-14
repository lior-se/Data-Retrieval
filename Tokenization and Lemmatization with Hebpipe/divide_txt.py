def split_file(file_path, lines_per_file, output_prefix):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i in range(0, len(lines), lines_per_file):
        with open(f'{output_prefix}_{i//lines_per_file + 1}.txt', 'w', encoding='utf-8') as output_file:
            output_file.writelines(lines[i:i + lines_per_file])

# Usage
file_path = 'A_text.txt'  # Path to your file
lines_per_file = 200  # Number of lines per split file
output_prefix = 'A_text_OWSout'  # Prefix for output files
split_file(file_path, lines_per_file, output_prefix)

file_path = 'B_text.txt'  # Path to your file
lines_per_file = 200  # Number of lines per split file
output_prefix = 'B_text_OWSout'  # Prefix for output files
split_file(file_path, lines_per_file, output_prefix)

file_path = 'C_text.txt'  # Path to your file
lines_per_file = 200  # Number of lines per split file
output_prefix = 'C_text_OWSout'  # Prefix for output files
split_file(file_path, lines_per_file, output_prefix)