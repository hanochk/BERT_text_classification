# This is a sample Python script.
import pandas as pd
# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.model_selection import *

from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, PreTrainedTokenizerBase, AdamW
import re
import torch
from torch import Tensor
import numpy as np
import time
from eval_metrices import *
from train_eval_transformer_classification import *

class TextCleaner():
    def __init__(self):
        pass

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text



class NewsDataset_token(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels

    def __getitem__(self, idx):
        item = dict()
        item['text'] = self.text[idx]

        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def add_padding_tokens(input_id_chunks: list[Tensor], mask_chunks: list[Tensor]) -> None:
    """Adds padding tokens (token id = 0) at the end to make sure that all chunks have exactly 512 tokens."""
    for i in range(len(input_id_chunks)):
        # get required padding length
        pad_len = 512 - input_id_chunks[i].shape[0]
        # check if tensor length satisfies required chunk size
        if pad_len > 0:
            # if padding length is more than 0, we must add padding
            input_id_chunks[i] = torch.cat([input_id_chunks[i], Tensor([0] * pad_len)])
            mask_chunks[i] = torch.cat([mask_chunks[i], Tensor([0] * pad_len)])

def stack_tokens_from_all_chunks(input_id_chunks: list[Tensor], mask_chunks: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Reshapes data to a form compatible with BERT model input."""
    input_ids = torch.stack(input_id_chunks)
    attention_mask = torch.stack(mask_chunks)

    return input_ids.long(), attention_mask.int()
def transform_single_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    stride: int,
    minimal_chunk_length: int,
    maximal_text_length: int,
) -> tuple[Tensor, Tensor]:
    """Transforms (the entire) text to model input of BERT model."""
    if maximal_text_length:
        tokens = tokenize_text_with_truncation(text, tokenizer, maximal_text_length)
    else:
        tokens = tokenize_whole_text(text, tokenizer)
    input_id_chunks, mask_chunks = split_tokens_into_smaller_chunks(tokens, chunk_size, stride, minimal_chunk_length)
    add_special_tokens_at_beginning_and_end(input_id_chunks, mask_chunks)
    add_padding_tokens(input_id_chunks, mask_chunks)
    input_ids, attention_mask = stack_tokens_from_all_chunks(input_id_chunks, mask_chunks)
    return input_ids, attention_mask


def main():
    # Use a breakpoint in the code line below to debug your script.
    # !python /content/drive/MyDrive/Colab\ Notebooks/BERT_text_classification/bert_text_classification.py
    colab = False
    if not colab:
        base_dir = r'C:\HanochWorkSpce\Projects\news_classification\BERT_text_classification'
    else:
        base_dir = '/content/drive/MyDrive/Colab Notebooks/BERT_text_classification'
    unique_run_name = str(int(time.time()))

    test_loader, train_loader, val_loader = data_processing(base_dir, debug=False)


    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    # Fine-tune only classiifer
    for name, param in model.named_parameters():
        if 'classifier' not in name:  # classifier layer
            param.requires_grad = False
    # Validation
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    n_epochs = 3
    lr = 1e-2
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print('Learning rate', lr)
    all_val_total_loss = list()
    all_train_total_loss = list()
    for epoch in range(n_epochs):
        all_targets, all_predictions, train_total_loss = train_model(model, train_loader=train_loader,
                                                                   optimizer=optimizer,
                                                                    device=device, num_epochs=1)

        all_train_total_loss.append(np.array(train_total_loss))

        all_targets_val, all_predictions_val, val_total_loss = eval_model(model, dataloader=val_loader,
                                                              device=device)

        all_val_total_loss.append(np.array(val_total_loss).mean())

    model.save_pretrained(os.path.join(base_dir, unique_run_name + 'distilbert-base-uncased-news_cls'), from_pt=True)

    print('Validation set AP : ')
    roc_plot(all_targets_val, all_predictions_val[:, 1], positive_label=1, save_dir=base_dir,
                        unique_id=unique_run_name + 'Validation set')

    plt.figure()
    plt.plot([x.mean() for x in all_train_total_loss ], 'b', label='train loss')
    plt.plot(all_val_total_loss, 'r', label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Learning curve epochs={}".format(n_epochs))
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig(
        os.path.join(base_dir, unique_run_name + 'learning_curve.jpg'))
    plt.close('all')

    all_targets_test, all_predictions_test, test_total_loss = eval_model(model, dataloader=test_loader,
                                                              device=device)

    roc_plot(all_targets_test, all_predictions_test[:, 1], positive_label=1, save_dir=base_dir,
                    unique_id=unique_run_name + 'Test set')
    print(all_predictions_test)
    print(all_targets_test)


def data_processing(base_dir: str, debug=True):
    df = pd.read_csv(os.path.join(base_dir, 'assignment_data_en.csv'), index_col=False)
    print('Web text typ', df.content_type.unique())
    print('evidence', len(df[df.content_type == 'news']) / len(df))


    if debug:
        print('partial data')
        df = df[:100]
    cleaner = TextCleaner()  # remove \n , un necessay characters , and lowercasing
    df['cleaned_text'] = df['scraped_text'].apply(cleaner.clean_text)
    df['label'] = df['content_type'].apply(lambda x: 1 if x == 'news' else 0)
    train_texts, test_texts, train_labels, test_labels = train_test_split(df['cleaned_text'],
                                                                          df['label'], test_size=.1, random_state=1)
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts,
                                                                        train_labels, test_size=.1, random_state=1)
    print('Train set {}, Val set {}, test set {}'.format(len(train_labels), len(val_labels), len(test_labels)))
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    # model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=2)
    if 1:
        # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        train_encodings = tokenizer(train_texts.to_list(), padding=True, truncation=True,
                                    return_tensors="pt")
        val_encodings = tokenizer(val_texts.to_list(), padding=True, truncation=True,
                                  return_tensors="pt")
        test_encodings = tokenizer(test_texts.to_list(), padding=True, truncation=True,
                                   return_tensors="pt")

        train_dataset = NewsDataset_token(train_encodings, train_labels.to_list())
        val_dataset = NewsDataset_token(val_encodings, val_labels.to_list())
        test_dataset = NewsDataset_token(test_encodings, test_labels.to_list())
    else:
        train_dataset = NewsDataset(train_texts.to_list(), train_labels)
        val_dataset = NewsDataset(val_texts.to_list(), val_labels)
        test_dataset = NewsDataset(test_texts.to_list(), test_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    return test_loader, train_loader, val_loader


if __name__ == '__main__':
    main()
