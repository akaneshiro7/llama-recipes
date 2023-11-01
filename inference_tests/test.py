# Get prompts
with open('./inference_tests/filtered_alpaca_test_set_1.txt', 'r') as f:
    instructions = [line.strip() for line in f]
    