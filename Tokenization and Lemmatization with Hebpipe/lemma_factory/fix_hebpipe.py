import os

def process_files_v2(base_dir, file_range=range(1,26)):
    """
    Revised function to process the files as per the updated requirement.

    Args:
    - base_dir (str): The base directory where the files are located.
    - file_range (range): The range of file numbers to process (default is 1 to 25).
    """
    for num in file_range:
        txt_file = os.path.join(base_dir, f"C_text_OWSout_{num}.txt")
        conllu_file = os.path.join(base_dir, f"C_text_OWSout_{num}.conllu")

        # Read the .txt file and split into lines
        with open(txt_file, 'r',encoding='utf-8') as f:
            txt_lines = [line.strip() for line in f.readlines()]

        # Process the .conllu file
        new_conllu_content = []
        skip_next_text = False
        txt_line_index = 0

        with open(conllu_file, 'r',encoding='utf-8') as f:
            lines = f.readlines()

            for i, line in enumerate(lines):
                line = line.strip()

                if line.startswith("# text ="):
                    if skip_next_text:
                        # Check if the line contains the remaining part of the txt_line
                        if txt_lines[txt_line_index].endswith(line.replace("# text =", "").strip()):
                            skip_next_text = False
                            txt_line_index += 1
                        continue
                    else:
                        if txt_line_index < len(txt_lines) and line == f"# text = {txt_lines[txt_line_index]}":
                            txt_line_index += 1
                        else:
                            skip_next_text = True

                new_conllu_content.append(line)

        # Write the modified .conllu file
        with open(conllu_file, 'w',encoding='utf-8') as f:
            for line in new_conllu_content:
                f.write(line + "\n")

# Note: The function won't be executed here as it requires access to the specific files.
# You should provide the correct path to your files when using this function.
process_files_v2("")
