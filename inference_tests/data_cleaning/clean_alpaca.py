import pandas as pd

df = pd.read_json('alpaca_data.json')

filtered = df[df['input'] == ""]

print(f"{len(df) - len(filtered)} instructions removed.")

instructions = filtered['instruction']
instructions.to_csv('filtered_alpaca.csv', index=False)

instructions = instructions.tolist()

