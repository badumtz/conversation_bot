
import torch
from torchtext.data import Field, TabularDataset, BucketIterator

input_tokenizer = Field(tokenize="spacy", lower=True, init_token="<sos>", eos_token="<eos>")
output_tokenizer = Field(tokenize="spacy", lower=True, init_token="<sos>", eos_token="<eos>")

fields = [('input', input_tokenizer), ('output', output_tokenizer)]

train_data, test_data = TabularDataset.splits(
    path='data', train='train.csv', test='test.csv', format='csv', fields=fields)

input_tokenizer.build_vocab(train_data, min_freq=2)
output_tokenizer.build_vocab(train_data, min_freq=2)

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), batch_size=64, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

lines_filepath = "C:/Users/stefa/PycharmProjects/pythonProject10/dataset/movie_lines.tsv"
conversations_filepath = "C:/Users/stefa/PycharmProjects/pythonProject10/dataset/movie_conversations.tsv"

with open(lines_filepath, 'r', encoding='utf-8', errors='ignore') as file:
    lines = file.readlines()

with open(conversations_filepath, 'r', encoding='utf-8', errors='ignore') as file:
    conversations = file.readlines()

input_messages = []
output_responses = []

for conversation in conversations:

    parts = conversation.strip().split(" +++$+++ ")

    line_ids_str = parts[-1]

    line_ids = [line_id.strip(" []'") for line_id in line_ids_str.split(",")]

    line_ids = [line_id for line_id in line_ids if line_id.startswith("L")]

    conversation_text = [lines[int(line_id[1:])-1].strip() for line_id in line_ids if 1 <= int(line_id[1:]) <= len(lines)]

    for i in range(len(conversation_text) - 1):
        input_messages.append(conversation_text[i])
        output_responses.append(conversation_text[i + 1])

for i in range(min(5, len(input_messages))):
    print("Input:", input_messages[i])
    print("Response:", output_responses[i])
    print()