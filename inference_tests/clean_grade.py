import pandas as pd

df = pd.read_json('grade_math_data.jsonl', lines=True)

prompts = df['question']

prompts.to_csv('filtered_grade.csv', index=False)