# This code defines the Decoder class, which is a subclass of the TensorFlow Keras Layer class. 
# The class has the following features:

# The constructor takes in the following parameters: the number of heads (nhead), the number of layers (num_layers), 
# the dimension of the feed-forward network (dim_feedforward), and the dimension of the model (d_model).
# it has an embedding layer, this layer will map the input word index to its embedding vector
# it has a positional encoding layer, this is used to add position information to the input embeddings
# it has a list of layers, each element of the list is an object of either MultiHeadAttention or FeedForward class.
# The call method applies the input through a series of layers. The input first passed through the embedding layer, 
# then positional encoding is added, and then the input is passed through the list of layers.
# The call method takes two inputs, the first one is the input sequence and the second one is the output of the encoder.
# It also returns the attention weights for each layer and each block.


import tensorflow as tf
from .layers import MultiHeadAttention, FeedForward

class Decoder(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_len, d_model)
        self.layers = [
            MultiHeadAttention(d_model, nhead)
            MultiHeadAttention(d_model, nhead)
            FeedForward(d_model, dim_feedforward)
            for _ in range(num_layers)
        ]

    def call(self, x, enc_output):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
        x += self.pos_encoding[:, :tf.shape(x)[1], :]
        attention_weights = {}
        for i in range(self.num_layers):
            x, block1, block2 = self.layers[i](x, enc_output)
            attention_weights[f'layer_{i}_block1'] = block1
            attention_weights[f'layer_{i}_block2'] = block2
        return x, attention_weights
