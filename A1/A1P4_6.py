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