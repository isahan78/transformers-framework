# This code defines a class FFN which takes in three parameters: 
# the dimension of the model (d_model), the dimension of the feed forward network (dff), and the dropout rate (dropout_rate).
# It has two dense layers, the first one with a relu activation function and the second one without any activation function. 
# The call method applies the linear_1 layer on the input, then applies dropout on the output of the linear_1 layer, 
# then applies the linear_2 layer on the output of the dropout, and then applies the layernorm on the final output.
# The dropout rate is set to 0.1 by default. The dropout layer is used to prevent overfitting by randomly dropping out some 
# neurons during training. Layer normalization is also used to normalize the output of the final dense layer, which helps 
# stabilize the training process.


import tensorflow as tf

class FFN(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super(FFN, self).__init__()
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.linear_1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.linear_2 = tf.keras.layers.Dense(d_model)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        x = self.linear_1(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.layernorm(x)
        return x

