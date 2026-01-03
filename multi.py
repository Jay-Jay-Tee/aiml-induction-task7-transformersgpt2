import numpy as np

class MultiHeadSelfAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_Q = np.random.randn(d_model, d_model)
        self.W_K = np.random.randn(d_model, d_model)
        self.W_V = np.random.randn(d_model, d_model)

        self.W_O = np.random.randn(d_model, d_model)

    def forward(self, X):
        # X has dimensions (No. of tokens,dimensions of model)

        T = X.shape[0]

        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V


        Q = Q.reshape(T, self.num_heads, self.d_head)
        K = K.reshape(T, self.num_heads, self.d_head)
        V = V.reshape(T, self.num_heads, self.d_head)

        heads = []

        for h in range(self.num_heads):
            q = Q[:, h]
            k = K[:, h]
            v = V[:, h]

            
            scores = (q @ k.T) / np.sqrt(self.d_head)

            
            mask = np.triu(np.ones((T, T)), k=1) * -1e9
            scores = scores + mask

            
            attention = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

            
            head_output = attention @ v
            heads.append(head_output)

        
        concat = np.concatenate(heads, axis=1)

        
        return concat @ self.W_O
