
import re
import pandas as pd

# Function to parse the text file
def parse_text_file(input_file, output_file):
 ground_truth_labels = []
 predicted_labels = []

 with open(input_file, 'r', encoding='utf-8') as file:
  content = file.read()  # Read the entire file content

# Use regex to find all occurrences of Ground Truth Label and Predicted Label
 matches = re.findall(r'Ground Truth Label: (\d+), Predicted Label: (.*)', content)

 for match in matches:
  ground_truth_labels.append(int(match[0]))  # Append Ground Truth Label
  predicted_labels.append(match[1])      # Append Predicted Label

# Create a DataFrame and save to CSV
  df = pd.DataFrame({
   'Ground Truth Label': ground_truth_labels,
   'Predicted Label': predicted_labels
  })

  df.to_csv(output_file, index=False, encoding='utf-8')
  print(f'Data has been saved to {output_file}')

# Specify the input and output files
input_file = 'D:\\Code\\LLMTraining-main\\data\\input.txt'  # Change to your input file name
output_file = 'D:\\Code\\LLMTraining-main\\data\\output.csv'  # Desired output file name

# Run the function
parse_text_file(input_file, output_file)