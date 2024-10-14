import pandas as pd
from tqdm import tqdm

def calculate_means_and_save(input_excel_path, output_excel_path):
    # Read the Excel file
    df = pd.read_excel(input_excel_path)
    print("1")

    # Calculate mean of each column
    column_means = tqdm(df.mean())
    print("2")

    # Sort the means in ascending order and select the 20 columns with the lowest means
    lowest_means = column_means.nsmallest(20)

    # Sort the means in descending order and select the 20 columns with the highest means
    highest_means = column_means.nlargest(20)

    # Combine the two series
    combined_series = pd.concat([highest_means, lowest_means], keys=['Highest Means', 'Lowest Means'])

    # Convert the combined series into a DataFrame for easier Excel output
    combined_df = combined_series.reset_index()
    combined_df.columns = ['Mean Type', 'Column Name', 'Mean']

    # Write the DataFrame to a new Excel file
    combined_df.to_csv(output_excel_path,encoding='utf-8', index=False)


# Example usage
input_excel_path = 'A_lemma.xlsx'
output_excel_path = 'A_lemma_20output.csv'
calculate_means_and_save(input_excel_path, output_excel_path)
