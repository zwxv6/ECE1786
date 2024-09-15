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

