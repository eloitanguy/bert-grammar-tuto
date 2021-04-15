import wget
import os
import pandas as pd
from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, random_split


def download():
    print('Downloading dataset...')

    # The URL for the dataset zip file.
    url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'

    # Download the file (if we haven't already)
    if not os.path.exists('./cola_public_1.1.zip'):
        wget.download(url, './cola_public_1.1.zip')


def encode_sentences(tokenizer, sentences):
    # Tokenize all of the sentences and map the tokens to their word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=64,  # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding with 0/1).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


def create_datasets():
    download()

    # Train + val
    df = pd.read_csv("./cola_public/raw/in_domain_train.tsv",
                     delimiter='\t',
                     header=None,
                     names=['sentence_source', 'label', 'label_notes', 'sentence'])

    sentences = df.sentence.values
    labels = df.label.values

    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    input_ids, attention_masks = encode_sentences(tokenizer, sentences)
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('Saving the datasets...')
    torch.save(train_dataset, 'data/train_dataset.pt')
    torch.save(val_dataset, 'data/val_dataset.pt')

    # test
    df = pd.read_csv("./cola_public/raw/out_of_domain_dev.tsv", delimiter='\t', header=None,
                     names=['sentence_source', 'label', 'label_notes', 'sentence'])
    sentences = df.sentence.values
    labels = torch.tensor(df.label.values)
    input_ids, attention_masks = encode_sentences(tokenizer, sentences)
    prediction_data = TensorDataset(input_ids, attention_masks, labels)

    torch.save(prediction_data, 'data/test_dataset.pt')


if __name__ == '__main__':
    create_datasets()
