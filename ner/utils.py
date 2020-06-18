import tensorflow as tf
import ner
import os
import yaml

def load_config(model_config_name='bert_ner_data_dist_kl_config.yaml', master_config_name='config.yaml'):
  with open(os.path.join(ner.CONFIG_DIR, master_config_name)) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
  with open(os.path.join(ner.CONFIG_DIR, model_config_name)) as file:
    model_config = yaml.load(file, Loader=yaml.FullLoader)

  config.update(model_config)
  return config

@tf.function
def reduce_mean_masked(tensor, mask, axis=-1, eps=1e-9):
  masked_tensor = tensor * mask
  seq_lens = tf.reduce_sum(mask, axis=axis)
  prevent_zero_div_mask = tf.cast(tf.math.equal(seq_lens, 0.0), tf.float32)
  seq_lens = seq_lens + eps * prevent_zero_div_mask
  return tf.reduce_sum(masked_tensor, axis=axis) / seq_lens

@tf.function
def masked_sparse_categorical_crossentropy(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

@tf.function
def get_accuracy_no_other_mask(y_true):
  not_other = tf.math.logical_not(tf.math.equal(y_true, 1))
  not_pad = tf.math.logical_not(tf.math.equal(y_true, 0))
  return tf.cast(tf.math.logical_and(not_other, not_pad), tf.int32)
