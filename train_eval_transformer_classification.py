import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
import torch
from torch import Tensor



def eval_model(model, dataloader, device, **kwargs):

    all_targets = list()
    all_predictions = list()
    all_total_loss = list()

    model.eval()

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels, return_dict=False) #  its default loss is the crossentropy
            loss = outputs[0] # return the CE out of the model

            all_targets.append(labels)
            labels.to(device).long()
            total_loss = outputs[0]

            predictions = torch.nn.functional.softmax(outputs[1], dim=1)
            all_predictions.append(predictions.detach().cpu().numpy())

            all_total_loss.append(total_loss.detach().cpu().numpy().item())

        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)

        return all_targets, all_predictions, all_total_loss

def train_model(model, train_loader: DataLoader, optimizer, device='cpu', num_epochs=1):
    all_targets = list()
    all_predictions = list()
    # max_iter = kwargs.pop('max_iterations', -1)
    all_total_loss = list()

    running_loss = 0.0
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        # permutation = torch.randperm(len(X_train))
        # for i in range(0,len(X_train), batch_size):
        for batch in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            # indices = permutation[i:i+batch_size]
            # batch_x, batch_y = X_train[indices].cuda(), Y_train[indices].cuda()
            all_targets.append(batch['labels'])

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels, return_dict=False) #  its default loss is the crossentropy
            # outputs = model(batch_x)
            loss = outputs[0] # return the CE out of the model
            # loss2 = criterion(outputs[1], labels)
            all_total_loss.append(loss.detach().cpu().numpy().item())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predictions = torch.nn.functional.softmax(outputs[1], dim=1)
            all_predictions.append(predictions.detach().cpu().numpy())


        print('{} epoch loss: {:3f} '.format(epoch + 1, running_loss / len(train_loader.dataset)))
        running_loss = 0.0
        # plt.savefig(os.path.join(base_dir, 'training_loss_sanity.png'))

    return all_targets, all_predictions, all_total_loss

# def train_model(model, train_loader, optimizer, device='cpu', num_epochs=1):
#     all_targets = list()
#     all_predictions = list()
#     # max_iter = kwargs.pop('max_iterations', -1)
#     all_total_loss = list()
#
#     running_loss = 0.0
#     model.to(device)
#     model.train()
#     for epoch in range(num_epochs):
#         # permutation = torch.randperm(len(X_train))
#         # for i in range(0,len(X_train), batch_size):
#         for batch in tqdm(train_loader):
#             optimizer.zero_grad()
#             # indices = permutation[i:i+batch_size]
#             # batch_x, batch_y = X_train[indices].cuda(), Y_train[indices].cuda()
#             all_targets.append(batch['labels'])
#
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)
#             outputs = model(input_ids, attention_mask=attention_mask, labels=labels, return_dict=False) #  its default loss is the crossentropy
#             # outputs = model(batch_x)
#             loss = outputs[0] # return the CE out of the model
#             # loss2 = criterion(outputs[1], labels)
#             all_total_loss.append(loss.detach().cpu().numpy().item())
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#
#             predictions = torch.nn.functional.softmax(outputs[1], dim=1)
#             all_predictions.append(predictions.detach().cpu().numpy())
#
#
#         print('{} epoch loss: {:3f} '.format(epoch + 1, running_loss / len(train_loader.dataset)))
#         running_loss = 0.0
#         # plt.savefig(os.path.join(base_dir, 'training_loss_sanity.png'))
#
#     return all_targets, all_predictions, all_total_loss
