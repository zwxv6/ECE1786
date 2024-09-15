%matplotlib inline
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from nltk.tokenize import sent_tokenize
import time
import torch
import spacy
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import nltk
nltk.download('punkt')
import math
# running time around 2 hours by using google colab pro
def train_sgns(textlist, window, embedding_size):
    # Set up a model with Skip-gram with negative sampling (predict context with word)
    # textlist: a list of strings
    epochs = 30
    batchSize = 4
    filtered_lemmas, w2i, i2w = prepare_texts(textlist)

    # Create Training Data
    X,T,Y = tokenize_and_preprocess_text(filtered_lemmas, w2i, window)

    # Split the training data
    X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(X,T,Y,test_size=0.2)

    # instantiate the network & set up the optimizer
    model = SkipGramNegativeSampling(vocab_size=len(w2i), embedding_size=embedding_size)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.00001)
    probabilityFunction = torch.nn.Sigmoid()

    temp = len(X_train)/batchSize
    temp = math.floor(temp)

    totalLoss = []
    totalValidationLoss = []
    for i in range(epochs):
      print(i)
      currentEpochLoss = 0
      currentEpochValidationLoss = 0
      mark = 0
      for f in range(temp):
        optimizer.zero_grad()
        currentBatchLoss = 0
        for j in range(batchSize):
          logits = model(x=torch.tensor(X_train[j+mark]),t=torch.tensor(T_train[j+mark]))
          probability = probabilityFunction(logits)
          probability = torch.maximum(probability, torch.tensor([1e-5]))
          probability = torch.minimum(probability, torch.tensor([0.99999]))
          if y_train[j+mark] == 1:
            currentLoss = -torch.log(probability)
          elif y_train[j+mark] == 0:
            currentLoss = -torch.log(1-probability)
          currentBatchLoss = currentBatchLoss + currentLoss

        currentBatchLoss = currentBatchLoss/batchSize
        currentEpochLoss = currentEpochLoss + currentBatchLoss
        currentBatchLoss.backward()
        optimizer.step()
        mark = mark + batchSize
      averageCurrentEpochLoss = currentEpochLoss/temp
      totalLoss.append(averageCurrentEpochLoss.item())

      for g in range(len(X_test)):
        logitsV = model(x=torch.tensor(X_test[g]),t=torch.tensor(T_test[g]))
        probabilityV = probabilityFunction(logitsV)
        probabilityV = torch.maximum(probabilityV, torch.tensor([1e-5]))
        probabilityV = torch.minimum(probabilityV, torch.tensor([0.99999]))
        if y_test[g] == 1:
          tempLoss = -torch.log(probabilityV)
        elif y_test[g] == 0:
          tempLoss = -torch.log(1-probabilityV)
        currentEpochValidationLoss = currentEpochValidationLoss + tempLoss
      currentEpochValidationLoss = currentEpochValidationLoss/len(X_test)
      totalValidationLoss.append(currentEpochValidationLoss.item())

    network = []
    network.append(model)
    network.append(totalLoss)
    network.append(totalValidationLoss)

    return network

def prepare_texts(text, min_frequency=3):

    # Get a callable object from spacy that processes the text - lemmatizes and determines part of speech

    nlp = spacy.load("en_core_web_sm")

    # Some text cleaning. Do it by sentence, and eliminate punctuation.
    lemmas = []
    for sent in sent_tokenize(text):  # sent_tokenize separates the sentences
        for tok in nlp(sent):         # nlp processes as in Part III
            if tok.pos_ not in ["PUNCT", "SPACE", "SYM", "NUM", "X"] and tok.lemma_ not in "[]|.,/?'\"+-=":
                lemmas.append(tok.lemma_)

    # Count the frequency of each lemmatized word
    freqs = Counter()  # word -> occurrence
    for w in lemmas:
        freqs[w] += 1

    vocab = list(freqs.items())  # List of (word, occurrence)
    vocab = sorted(vocab, key=lambda item: item[1], reverse=True)  # Sort by decreasing frequency
    print(vocab)
    print(len(vocab))
    # per Mikolov, don't use the infrequent words, as there isn't much to learn in that case

    frequent_vocab = list(filter(lambda item: item[1]>=min_frequency, vocab))
    print(frequent_vocab)

    # Create the dictionaries to go from word to index or vice-verse

    w2i = {w[0]:i for i,w in enumerate(frequent_vocab)}
    i2w = {i:w[0] for i,w in enumerate(frequent_vocab)}

    # Create an Out Of Vocabulary (oov) token as well
    w2i["<oov>"] = len(frequent_vocab)
    i2w[len(frequent_vocab)] = "<oov>"

    # Set all of the words not included in vocabulary nuas oov
    filtered_lemmas = []
    for lem in lemmas:
        if lem not in w2i:
            filtered_lemmas.append("<oov>")
        else:
            filtered_lemmas.append(lem)

    return filtered_lemmas, w2i, i2w

f = open("/content/LargerCorpus.txt")
lines = f.read()
f.close()
fl, wi, iw = prepare_texts(lines)

def tokenize_and_preprocess_text(textlist, w2i, window):
    """
    Skip-gram negative sampling: Predict if the target word is in the context.
    Uses binary prediction so we need both positive and negative samples
    """
    X, T, Y = [], [], []
    neighbor = (window - 1) / 2

    # Tokenize the input

    # TO DO

    n = 0
    for word in textlist:
        for i in range(1, int(neighbor) + 1):
            if n - i >= 0:
                X.append(w2i[word])
                T.append(w2i[textlist[n - i]])
                Y.append(1)
                X.append(w2i[word])
                T.append(np.random.randint(0, len(w2i)))
                Y.append(0)
            if n + i < len(textlist):
                X.append(w2i[word])
                T.append(w2i[textlist[n + i]])
                Y.append(1)
                X.append(w2i[word])
                T.append(np.random.randint(0, len(w2i)))
                Y.append(0)
        n = n + 1
    # Loop through each token

    # TO DO

    return X, T, Y

class SkipGramNegativeSampling(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()

        # TO DO
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embeddingA = torch.nn.Embedding(self.vocab_size, self.embedding_size)
        self.embeddingB = torch.nn.Embedding(self.vocab_size, self.embedding_size)

        self.embeddingA.weight.data.uniform_(-0.5, 0.5)
        self.embeddingB.weight.data.uniform_(-0.5, 0.5)

    def forward(self, x, t):

        # x: torch.tensor of shape (batch_size), context word
        # t: torch.tensor of shape (batch_size), target ("output") word.
        # TO DO
        embeddingX = self.embeddingA(x)
        embeddingT = self.embeddingB(t)
        prediction = torch.dot(embeddingX, embeddingT)

        return prediction

network = train_sgns(lines,5,8)
embedding = network[0]
totalLoss = network[1]
totalValidationLoss = network[2]
plt.plot(totalLoss)
plt.plot(totalValidationLoss)