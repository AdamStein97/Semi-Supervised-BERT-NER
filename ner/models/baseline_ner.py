import tensorflow as tf


class BaselineNER(tf.keras.Model):
    def __init__(self, mlp_dims=None, mlp_activation=tf.nn.relu, vocab_size=30000, embed_size=784,
                 latent_dim=32, categories=10, rate=0.0, **kwargs):
        super(BaselineNER, self).__init__()

        self.embed_layer = tf.keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)

        if mlp_dims is None:
            mlp_dims = [256, 128, 64]

        self.mlp = []
        for dim in mlp_dims:
            self.mlp += [tf.keras.layers.Dense(dim, activation=mlp_activation), tf.keras.layers.Dropout(rate)]

        self.encode_layer = tf.keras.layers.Dense(latent_dim)
        self.categorise_layer = tf.keras.layers.Dense(categories, activation='softmax')

    def call(self, input_word_ids, **kwargs):
        sequence_output = self.embed_layer(input_word_ids)
        for layer in self.mlp:
            sequence_output = layer(sequence_output)
        enc = self.encode_layer(sequence_output)
        return enc, self.categorise_layer(enc)