import tensorflow as tf
import tensorflow_hub as hub

class BERT_NER(tf.keras.Model):
    def __init__(self, mlp_dims=None, mlp_activation=tf.nn.relu, latent_dim=32, categories=10, bert_trainable=False,
                 rate=0.0, **kwargs):
        super(BERT_NER, self).__init__()
        self.bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                         trainable=bert_trainable)

        if mlp_dims is None:
            mlp_dims = [256, 128, 64]

        self.mlp = []
        for dim in mlp_dims:
            self.mlp += [tf.keras.layers.Dense(dim, activation=mlp_activation), tf.keras.layers.Dropout(rate)]

        self.encode_layer = tf.keras.layers.Dense(latent_dim)
        self.categorise_layer = tf.keras.layers.Dense(categories, activation='softmax')

    def call(self, input_word_ids, segment_ids, input_mask):
        _, sequence_output = self.bert_layer([input_word_ids, input_mask, segment_ids])
        for layer in self.mlp:
            sequence_output = layer(sequence_output)
        enc = self.encode_layer(sequence_output)
        return enc, self.categorise_layer(enc)