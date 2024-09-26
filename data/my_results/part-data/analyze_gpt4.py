import pandas as pd
import json

# Define file paths for the CSV files
file_path1 = 'data/my_results/gpt_4o_mini.csv'
file_path2 = 'data/my_results/gpt_4o_mini_kg_rag.csv'
file_path3 = 'data/my_results/gpt_4o_mini_kg_rag_knowledge.csv'
file_path4 = 'data/my_results/gpt_4o_mini_kg_rag_json.csv'
file_path5 = 'data/my_results/gpt_4o_mini_kg_rag_knowledge_json.csv'

# file_path1 = 'data/my_results/gemini_1.5_flash_kg_rag_based_mcq_from_monarch_and_robokop_response_0.csv'
# file_path2 = 'data/my_results/gemini_1.5_flash_kg_rag_based_mcq_from_monarch_and_robokop_response_1.csv'
# file_path3 = 'data/my_results/gemini_1.5_flash_kg_rag_based_mcq_from_monarch_and_robokop_response_2.csv'
# file_path4 = 'data/my_results/gemini_1.5_flash_kg_rag_based_mcq_from_monarch_and_robokop_response_3.csv'

# Load the CSV files into DataFrames
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)
df3 = pd.read_csv(file_path3)
df4 = pd.read_csv(file_path4)
df5 = pd.read_csv(file_path5)

# Define a function to check if the correct answer is present in the LLM answer
def contains_correct_answer(row):
    try: 
        return row['correct_answer'] == json.loads(row['llm_answer'].replace('```', '').replace('\n', '').replace('json', '').replace('{{', '{').replace('}}', '}').split('}')[0] + '}')['answer']
    except:
        return False

# Apply the function to each row of the DataFrames
df1['is_correct'] = df1.apply(contains_correct_answer, axis=1)
df2['is_correct'] = df2.apply(contains_correct_answer, axis=1)
df3['is_correct'] = df3.apply(contains_correct_answer, axis=1)
df4['is_correct'] = df4.apply(contains_correct_answer, axis=1)
df5['is_correct'] = df5.apply(contains_correct_answer, axis=1)

# Calculate the percentage of correct answers
correct_rate1 = df1['is_correct'].mean() * 100
correct_rate2 = df2['is_correct'].mean() * 100
correct_rate3 = df3['is_correct'].mean() * 100
correct_rate4 = df4['is_correct'].mean() * 100
correct_rate5 = df5['is_correct'].mean() * 100

print(f"Correct Answer Rate for {file_path1}: {correct_rate1:.2f}%")
print(f"Correct Answer Rate for {file_path2}: {correct_rate2:.2f}%")
print(f"Correct Answer Rate for {file_path3}: {correct_rate3:.2f}%")
print(f"Correct Answer Rate for {file_path4}: {correct_rate4:.2f}%")
print(f"Correct Answer Rate for {file_path5}: {correct_rate5:.2f}%")

# # Create a summary DataFrame with the results
# summary_df = pd.DataFrame({
#     'File': ['gpt_4o_mini.csv', 'gpt_4o_mini_kg_rag.csv', 'gpt_4o_mini_kg_rag_knowledge.csv', 'gpt_4o_mini_kg_rag_json.csv','gpt_4o_mini_kg_rag_knowledge_json.csv'],
#     # 'Correct Answer Rate (%)': [correct_rate1, correct_rate2, correct_rate3, correct_rate4, correct_rate5],
#     'Correct Answer Rate (%)': [correct_rate1, correct_rate2, correct_rate3, correct_rate4],
# })

# # Print the summary DataFrame
# print("Correct Answer Rates for Each CSV File:")
# print(summary_df)

# # Optionally, save the summary to a CSV file
# summary_df.to_csv('gpt_4o_analysis_summary.csv', index=False)
