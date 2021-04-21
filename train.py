import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import requests
import json
from utils import tokenize, stem, bag_of_words
from model import NeuralNet
import datetime

from aws import upload


class ChatDataset(Dataset):

    def __init__(self, X_train, Y_train):

        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


def train(batch_size, hidden_size, learning_rate, epochs, data, user_id, hidden_layers):
    all_words = []
    tags = []
    xy = []
    for intent in data['intents']:
        tags.append(intent['tag'])

        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, intent['tag']))

    all_words = [stem(word) for word in all_words]
    all_words = sorted(set(all_words))

    X_train = []
    Y_train = []

    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        Y_train.append(tags.index(tag))

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    dataset = ChatDataset(X_train=X_train, Y_train=Y_train)

    train_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    input_size = len(all_words)
    output_size = len(tags)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size, hidden_layers).to(device)

    #loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss = 0
    for epoch in range(epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(words)
            loss = criterion(outputs, labels)

            # backward and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) %100 ==0:
            print(f'epoch {epoch+1}/{epochs},loss={loss.item():.4f}')
    print(f'final lost, loss={loss.item():.4f}')

    d = {
    "model_state": model.state_dict(),
    "input_size":input_size,
    "output_size":output_size,
    "hidden_size":hidden_size,
     "all_words":all_words,
     "tags": tags,
     "data": data,
     "hidden_layers": hidden_layers
    }

    FILE = user_id+"_"+datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")+"_data.pth"
    torch.save(d, FILE)
    print(model)
    #upload to s3
    upload(FILE,user_id,loss.item())
    return model

# url = 'https://raw.githubusercontent.com/python-engineer/pytorch-chatbot/master/intents.json'
# data = requests.get(url)

# data = json.loads(data.text)
# train(8,8,0.001,1000,data)