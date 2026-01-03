import numpy as np

class PositionalEncoding:
    def __init__(self, max_len, d_model):
        self.pos_embedding = np.random.randn(max_len, d_model)

    def forward(self, token_embeddings):
        # token_embeddings has shape (No. of tokens, dimensions of model)

        T = token_embeddings.shape[0]

        pos_emb = self.pos_embedding[:T]

        return token_embeddings + pos_emb # goes as input to transformer blocks