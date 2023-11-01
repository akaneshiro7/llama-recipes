import random
import argparse
import pandas as pd

# Create the parser
parser = argparse.ArgumentParser(description='Process command line arguments.')
# Add the --file_name argument
parser.add_argument('--prompts', type=str, required=True, help='The name of the file')
# Add the --size argument
parser.add_argument('--size', type=int, required=True, help='The size parameter')
# Add the --seed argument
parser.add_argument('--seed', type=int, required=True, help='The seed value')
# Add the --output_file argument
parser.add_argument('--output_file', type=str, required=True, help='The name of the output file')

# Parse the arguments
args = parser.parse_args()
random.seed(args.seed)
instructions = pd.read_csv(args.prompts)
instructions = instructions.values.tolist()

randomInstructions = random.sample(instructions, args.size)

batch_size = args.size // 2

batches = []

with open(args.output_file + "_1.txt", 'w', encoding="utf-8") as f:
    items = randomInstructions[:batch_size]
    for i in items:
        item_without_newlines = i[0].replace('\n', ' ') 
        f.write(f"{item_without_newlines}\n")

with open(args.output_file + "_2.txt", 'w', encoding="utf-8") as f:
    items = randomInstructions[batch_size:]
    for i in items:
        item_without_newlines = i[0].replace('\n', ' ',) 
        f.write(f"{item_without_newlines}\n")