from collections import Counter
import numpy as np
import torch
import spacy
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt

f = open("/content/SmallSimpleCorpus.txt")
lines = f.read()
f.close()
l, vi, iv = prepare_texts(lines)

def train_word2vec(textlist, window, embedding_size):
    # Set up a model with Skip-gram (predict context with word)
    # textlist: a list of the strings
    epochs = 50
    batchSize = 4
    lemmas, v2i, i2v = prepare_texts(lines)

    # Create the training data
    # TO DO
    X, Y = tokenize_and_preprocess_text(lemmas, v2i, window)

    # Split the training data
    # TO DO
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # instantiate the network & set up the optimizer
    # TO DO
    model = Word2vecModel(vocab_size=len(v2i), embedding_size=embedding_size)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.005)
    lossFunction = torch.nn.CrossEntropyLoss()

    temp = len(X_train) / batchSize
    temp = math.floor(temp)
    # training loop
    # TO DO
    totalLoss = []
    totalValidationLoss = []
    # mark = 0
    for i in range(epochs):
        currentEpochLoss = 0
        currentEpochValidationLoss = 0

        mark = 0
        for f in range(temp):
            optimizer.zero_grad()
            logits, e = model(x=torch.tensor(X_train[mark:mark + batchSize]))
            currentLoss = lossFunction(logits, torch.tensor(y_train[mark:mark + batchSize]))
            currentEpochLoss = currentEpochLoss + currentLoss
            currentLoss.backward()
            optimizer.step()
            mark = mark + batchSize

        averageCurrentEpochLoss = currentEpochLoss / temp
        totalLoss.append(averageCurrentEpochLoss.item())

        logits0, e0 = model(x=torch.tensor(X_test))
        currentEpochValidationLoss = lossFunction(logits0, torch.tensor(y_test))
        totalValidationLoss.append(currentEpochValidationLoss.item())

    network = []
    network.append(model)
    network.append(totalLoss)
    network.append(totalValidationLoss)

    return network

def tokenize_and_preprocess_text(textlist, v2i, window):

    # Predict context with word. Sample the context within a window size.

    X, Y = [], []  # is the list of training/test samples

    # TO DO - create all the X,Y pairs
    neighbor = (window - 1) / 2

    n = 0
    for word in textlist:
      for i in range(1,int(neighbor)+1):
        if n - i >= 0:
          X.append(v2i[word])
          Y.append(v2i[textlist[n-i]])
        if n + i < len(textlist):
          X.append(v2i[word])
          Y.append(v2i[textlist[n+i]])
      n = n + 1

    return X, Y

class Word2vecModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        # initialize word vectors to random numbers

        #TO DO
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_size)
        # prediction function takes embedding as input, and predicts which word in vocabulary as output

        #TO DO
        self.embedding.weight.data.uniform_(-1, 1)

        self.out = torch.nn.Linear(self.embedding_size, self.vocab_size)


    def forward(self, x):
        """
        x: torch.tensor of shape (bsz), bsz is the batch size
        """
        #TO DO
        e = self.embedding(x)
        logits = self.out(e)
        return logits, e



network = train_word2vec(lines,5,2)
embedding = network[0]
totalLoss = network[1]
totalValidationLoss = network[2]
plt.plot(totalLoss)
plt.plot(totalValidationLoss)