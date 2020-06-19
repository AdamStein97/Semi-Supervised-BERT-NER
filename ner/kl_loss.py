import tensorflow as tf
from ner.utils import reduce_mean_masked

@tf.function
def get_flattened_label_distribution(model, batch, BATCH_SIZE=128, max_seq_length=50, categories=10, **kwargs):
  mask = batch['mask']
  _, pred_y = model(batch['word_id'], batch['segment_id'], mask)
  broadcast_mask = tf.cast(tf.broadcast_to(tf.expand_dims(mask, axis=-1), [BATCH_SIZE, max_seq_length, categories]),
                           tf.float32)

  masked_pred_y = pred_y * broadcast_mask
  q_ij = tf.reshape(masked_pred_y, [BATCH_SIZE * max_seq_length, categories])
  return q_ij

@tf.function
def get_reduced_label_distribution(model, batch, **kwargs):
  q_ij = get_flattened_label_distribution(model, batch, **kwargs)
  q_ij_mask = tf.expand_dims(tf.cast(tf.logical_not(tf.math.equal(tf.reduce_sum(q_ij, axis=-1), 0.0)), tf.float32), axis=-1)
  q_j = reduce_mean_masked(q_ij, q_ij_mask, axis=0)
  return q_j

@tf.function
def get_cluster_strength_prior(qij, eps=1e-9):
  f_j = tf.reduce_sum(qij, axis=0)
  qij_squared = qij ** 2
  qij_squared_normed = qij_squared / (f_j + eps)
  pij = qij_squared_normed / tf.reduce_sum(qij_squared_normed + eps, axis=1, keepdims=True)
  return pij

@tf.function
def calculate_cluster_strength_kl_loss(qij, eps=1e-9):
  pij = get_cluster_strength_prior(qij)
  kl = pij * tf.math.log(pij / (qij + eps) + eps)
  return tf.reduce_sum(kl)

@tf.function
def calculate_data_distribution_kl_loss(prior_data_distriburion, q_j, eps=1e-9):
  kl_loss = tf.reduce_sum(prior_data_distriburion * tf.math.log(prior_data_distriburion / q_j + eps))
  return kl_loss
