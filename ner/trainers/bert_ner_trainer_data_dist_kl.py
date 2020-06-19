import tensorflow as tf
import time
import os
import ner
from ner.models.bert_ner import BERT_NER
from ner.utils import masked_sparse_categorical_crossentropy, get_accuracy_no_other_mask
from ner.kl_loss import get_reduced_label_distribution, calculate_data_distribution_kl_loss

class BERT_NER_TraininerDataDistKL():
    def __init__(self, label_distribution=None, word_id_field='word_id', mask_field='mask', segment_id_field='segment_id', tag_id_field='tag_id', **kwargs):
        self.word_id_field = word_id_field
        self.mask_field = mask_field
        self.segment_id_field = segment_id_field
        self.tag_id_field = tag_id_field

        self.train_loss_ce = tf.keras.metrics.Mean(name='train_loss_ce')
        self.train_loss_kl = tf.keras.metrics.Mean(name='train_loss_kl')
        self.train_accuracy = tf.keras.metrics.Accuracy(name='train_acc')
        self.train_accuracy_no_other = tf.keras.metrics.Accuracy(name='train_acc_no_other')

        self.val_loss_ce = tf.keras.metrics.Mean(name='val_loss_ce')
        self.val_accuracy = tf.keras.metrics.Accuracy(name='val_acc')
        self.val_accuracy_no_other = tf.keras.metrics.Accuracy(name='val_acc_no_other')


        if label_distribution is None:
            label_distribution = [0.0, 8.468e-01, 4.300e-02, 1.530e-02, 3.270e-02, 3.520e-02, 2.560e-02,
                                           7.000e-04, 2.000e-04, 5.000e-04]

        self.label_distribution = tf.constant(label_distribution, dtype=tf.float32)

    @tf.function
    def train_ner_step(self, x, model, ner_optimizer):
        label = x[self.tag_id_field]
        with tf.GradientTape() as tape:
            _, pred_y = model(x[self.word_id_field], x[self.segment_id_field], x[self.mask_field])

            loss = masked_sparse_categorical_crossentropy(label, pred_y)

        gradients = tape.gradient(loss, model.trainable_variables)
        ner_optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        self.train_loss_ce(loss)

        pred_labels = tf.argmax(pred_y, axis=-1)

        self.train_accuracy(label, pred_labels, sample_weight=x[self.mask_field])
        self.train_accuracy_no_other(label, pred_labels, sample_weight=get_accuracy_no_other_mask(label))

    @tf.function
    def train_kl_step(self, x, model, kl_optimizer, **kwargs):
        with tf.GradientTape() as tape:
            q_j = get_reduced_label_distribution(model, x, **kwargs)

            loss = calculate_data_distribution_kl_loss(self.label_distribution, q_j)

        gradients = tape.gradient(loss, model.trainable_variables)
        kl_optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        self.train_loss_kl(loss)

    @tf.function
    def eval_ner_step(self, x, model):
        label = x[self.tag_id_field]
        _, pred_y = model(x[self.word_id_field], x[self.segment_id_field], x[self.mask_field])

        loss = masked_sparse_categorical_crossentropy(label, pred_y)
        self.val_loss_ce(loss)
        pred_labels = tf.argmax(pred_y, axis=-1)
        self.val_accuracy(label, pred_labels, sample_weight=x[self.mask_field])
        self.val_accuracy_no_other(label, pred_labels, sample_weight=get_accuracy_no_other_mask(label))

    def train_loop(self, model, ner_optimizer, kl_optimizer, labelled_ds, test_ds, unlabelled_ds,
                   EPOCHS=4, batches_per_epoch=10, model_save_weights_name='BERT_NER_data_dist_kl', **kwargs):

        for epoch in range(EPOCHS):
            start = time.time()
            for batch in range(batches_per_epoch):
                for x in unlabelled_ds.take(1):
                    self.train_kl_step(x, model, kl_optimizer, **kwargs)

                for x in labelled_ds.take(1):
                    self.train_ner_step(x, model, ner_optimizer)

                if batch % 3 == 0:
                    print(
                        'Batch {} CE Loss {:.4f} KL Loss {:.4f} \n Accuracy {:.4f}  Accuracy No "O" {:.4f}'.format(
                            batch, self.train_loss_ce.result(), self.train_loss_kl.result(),
                            self.train_accuracy.result(), self.train_accuracy_no_other.result()))

            print(
                'Epoch {} CE Loss {:.4f} KL Loss {:.4f} \n Accuracy {:.4f}  Accuracy No "O" {:.4f}'.format(
                    epoch + 1, self.train_loss_ce.result(), self.train_loss_kl.result(),
                    self.train_accuracy.result(), self.train_accuracy_no_other.result()))

            for x in test_ds:
                self.eval_ner_step(x, model)

            print('Epoch {} Val Loss {:.4f} Val Accuracy {:.4f} Accuracy No "O" {:.4f}'.format(
                epoch + 1, self.val_loss_ce.result(), self.val_accuracy.result(), self.val_accuracy_no_other.result()))

            self.train_loss_ce.reset_states()
            self.train_loss_kl.reset_states()
            self.train_accuracy.reset_states()
            self.train_accuracy_no_other.reset_states()
            self.val_loss_ce.reset_states()
            self.val_accuracy.reset_states()
            self.val_accuracy_no_other.reset_states()

            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

            model.save_weights(os.path.join(ner.MODEL_DIR, model_save_weights_name))

        return model


    def train(self, model=None, model_start_weights_filename='BERT_NER_final',
              ner_lr=1e-3, kl_lr=1e-3, **kwargs):

        if model is None:
            model = BERT_NER(**kwargs)

        ner_optimizer = tf.keras.optimizers.Adam(ner_lr)
        kl_optimizer = tf.keras.optimizers.Adam(kl_lr)

        if model_start_weights_filename is not None:
            model.load_weights(os.path.join(ner.MODEL_DIR, model_start_weights_filename))

        trained_model = self.train_loop(model,ner_optimizer, kl_optimizer, **kwargs)

        return trained_model

