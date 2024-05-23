# This is a sample Python script.
import pandas as pd
# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split

from transformers import get_linear_schedule_with_warmup
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
import re
import torch
import torch
from torch import Tensor

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
    maximal_text_length: Optional[int],
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

def main_1(name):

    train_texts, train_labels = read_imdb_split('../input/aclmdb/aclImdb/train')
    test_texts, test_labels = read_imdb_split('../input/aclmdb/aclImdb/test')
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    train_dataset = NewsDataset(train_encodings, train_labels)
    val_dataset = NewsDataset(val_encodings, val_labels)
    test_dataset = NewsDataset(test_encodings, test_labels)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        for batch in tqdm(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()
            optim.zero_grad()
    model.eval()

def main():
    # Use a breakpoint in the code line below to debug your script.
    base_dir = r'C:\HanochWorkSpce\Projects\news_classification\assignment_data_en.csv'
    df = pd.read_csv(base_dir, index_col=False)
    print('Web text typ', df.content_type.unique())
    print('evidence', len(df[df.content_type=='news'])/len(df))

    cleaner = TextCleaner() # remove \n , un necessay characters , and lowercasing
    df['cleaned_text'] = df['scraped_text'].apply(cleaner.clean_text)
    df['label'] = df['content_type'].apply(lambda x: 1 if x=='news' else 0)
    train_texts, test_texts, train_labels, test_labels = train_test_split(df['cleaned_text'],
                                                        df['label'], test_size=.1)

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts,
                                                        train_labels, test_size=.1)

    print('Train set {}, Val set {}, test set {}'.format(len(train_labels), len(val_labels), len(test_labels)))
    # tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # train_encodings = tokenizer(train_texts.to_list(), padding=True, truncation=True, return_tensors="pt")
    # val_encodings = tokenizer(val_texts.to_list(), padding=True, truncation=True)
    # test_encodings = tokenizer(test_texts.to_list(), padding=True, truncation=True)

    # train_dataset = NewsDataset(train_encodings, train_labels)
    # val_dataset = NewsDataset(val_encodings, val_labels)
    # test_dataset = NewsDataset(test_encodings, test_labels)

    train_dataset = NewsDataset(train_texts.to_list(), train_labels)
    val_dataset = NewsDataset(val_texts, val_labels)
    test_dataset = NewsDataset(test_texts, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    n_epochs = 5
    batch_size = 1

    bert_distil = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(bert_distil.parameters(), lr=1e-3)

    # X_train = []
    # Y_train = []
    # for row in train_df.iterrows():
    #     seq = tokenizer.encode(row[1]['text'],  add_special_tokens=True, pad_to_max_length=True)[:100]
    #     X_train.append(torch.tensor(seq).unsqueeze(0))
    #     Y_train.append(torch.tensor([row[1]['target']]))
    # X_train = torch.cat(X_train)
    # Y_train = torch.cat(Y_train)

    running_loss = 0.0
    bert_distil.cuda()
    bert_distil.train(True)
    for epoch in range(n_epochs):
        # permutation = torch.randperm(len(X_train))
        # for i in range(0,len(X_train), batch_size):
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            # indices = permutation[i:i+batch_size]
            # batch_x, batch_y = X_train[indices].cuda(), Y_train[indices].cuda()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            # outputs = bert_distil(batch_x)
            loss = criterion(outputs[0], labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('[%d] epoch loss: %.3f' %
          (epoch + 1, running_loss / len(X_train) * batch_size))
        running_loss = 0.0

if __name__ == '__main__':
    main()
