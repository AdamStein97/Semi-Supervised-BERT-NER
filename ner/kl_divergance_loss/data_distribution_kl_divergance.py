import tensorflow as tf
from ner.utils import reduce_mean_masked

@tf.function
def get_unlabelled_data_distribution(model, unlabelled_batch, eps=1e-9, BATCH_SIZE=128, max_seq_length=50, categories=10, **kwargs):
  normalise_mask = tf.zeros((BATCH_SIZE, max_seq_length, categories)) + eps
  unlabelled_softmax = tf.zeros((BATCH_SIZE, max_seq_length, categories))
  mask = unlabelled_batch['mask']
  _, unlabelled_pred_y = model(unlabelled_batch['word_id'], unlabelled_batch['segment_id'], mask)
  broadcast_mask = tf.cast(tf.broadcast_to(tf.expand_dims(mask, axis=-1), [BATCH_SIZE, max_seq_length, categories]), tf.float32)
  unlabelled_softmax += unlabelled_pred_y * broadcast_mask
  normalise_mask += broadcast_mask

  mean_unlabelled_softmax = unlabelled_softmax / normalise_mask
  q_ij = tf.reshape(mean_unlabelled_softmax, [BATCH_SIZE*max_seq_length, categories])
  q_ij_mask = tf.expand_dims(tf.cast(tf.logical_not(tf.math.equal(tf.reduce_sum(q_ij, axis=-1), 0.0)), tf.float32), axis=-1)
  q_j = reduce_mean_masked(q_ij, q_ij_mask, axis=0)
  return q_j

@tf.function
def calculate_data_distribution_kl_loss(prior_data_distriburion, q_j, eps=1e-9):
  # prior_data_distriburion = tf.constant(label_distribution, dtype=tf.float32)
  kl_loss = tf.reduce_sum(prior_data_distriburion * tf.math.log(prior_data_distriburion / q_j + eps))
  return kl_loss