import tensorflow_hub as hub
import bert
import pandas as pd
import ner
import os
import tensorflow as tf

class Preprocessor():
    def __init__(self, max_seq_length=50, **kwargs):
        bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                    trainable=False)
        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        self.tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file)
        self.max_seq_length = max_seq_length
        self.sep_word_id = self.tokenizer.convert_tokens_to_ids(["[SEP]"])[0]


    def get_word_id(self, word):
        t = self.tokenizer.tokenize(word)
        if len(t) == 0:
            return 0
        else:
            t_id = self.tokenizer.convert_tokens_to_ids([t[0]])
            return t_id[0]

    def _create_padded_frame(self, data, sentence_id_field='sentence_id', word_field='Word', word_id_field='word_id',
                             tag_id_field='tag_id', **kwargs):

        cls_word_id = self.tokenizer.convert_tokens_to_ids(["[CLS]"])[0]

        sentence_numbers = data.index.unique()
        full_df = pd.DataFrame()
        for sentence_number in sentence_numbers:
            sentence_df = data.loc[sentence_number][[sentence_id_field, word_field, word_id_field, tag_id_field]]
            sentence_df = pd.DataFrame(
                {sentence_id_field: [sentence_number], word_field: ['[CLS]'], word_id_field: [cls_word_id],
                 tag_id_field: [0]}).append(
                sentence_df, ignore_index=True)
            sentence_df = sentence_df.append(pd.DataFrame(
                {sentence_id_field: [sentence_number], word_field: ['[SEP]'], word_id_field: [self.sep_word_id], tag_id_field: [0]}),
                ignore_index=True)
            pad_df_len = self.max_seq_length - len(sentence_df)
            if pad_df_len > 0:
                pad_df = pd.DataFrame({sentence_id_field: [sentence_number] * pad_df_len, word_field: ['[PAD]'] * pad_df_len,
                                       word_id_field: [0] * pad_df_len, tag_id_field: [0] * pad_df_len})
                padded_df = sentence_df.append(pad_df, ignore_index=True)
                full_df = full_df.append(padded_df, ignore_index=True)

        return full_df

    def preprocess_data(self, csv_data_file="raw_ner_dataset.csv", csv_encoding='unicode_escape', csv_save_filename='preprocessed_ner_dataset.csv',
                        id2tag=None, tag_field='Tag', sentence_number_field='Sentence #', word_field='Word',
                        tag_preprocess=lambda x: x.split('-')[-1], sentence_id_preprocess=lambda x: int(x.split(": ")[-1]), **kwargs):

        if id2tag is None:
            id2tag = {
                0: 'PAD',
                1: 'O',
                2: 'geo',
                3: 'gpe',
                4: 'per',
                5: 'org',
                6: 'tim',
                7: 'art',
                8: 'nat',
                9: 'eve'}

        data = pd.read_csv(os.path.join(ner.DATA_DIR,csv_data_file), encoding=csv_encoding)

        data['preprocessed_tag'] = data[tag_field].apply(tag_preprocess)
        data['sentence_id'] = data[sentence_number_field].ffill().apply(sentence_id_preprocess)
        data['word_id'] = data[word_field].apply(self.get_word_id)
        data.index = data['sentence_id']

        tag2id = {v: k for k, v in id2tag.items()}

        data['tag_id'] = data['preprocessed_tag'].apply(lambda x: tag2id[x])

        padded_df = self._create_padded_frame(data)

        padded_df.to_csv(os.path.join(ner.DATA_DIR, csv_save_filename), index=False)


    def _get_segments(self, example, word_id_field='word_id', mask_field='mask', segment_id_field='segment_id', **kwargs):
        tokens = example[word_id_field]
        seq_len = tf.size(tokens)
        segment_breaks = tf.where(tf.math.equal(tokens, self.sep_word_id)) + 1
        segment_embed = tf.zeros(seq_len, dtype=tf.int32)
        start_segment = tf.constant(0, shape=(1,))
        for i in range(tf.shape(segment_breaks)[0]):
            segment_length = tf.cast(segment_breaks[i], tf.int32) - start_segment
            end_pad_len = seq_len - segment_length - start_segment
            segment_embed += tf.concat(
                [tf.zeros(start_segment, dtype=tf.int32), tf.ones(segment_length, dtype=tf.int32) * i,
                 tf.zeros(end_pad_len, dtype=tf.int32)], axis=-1)
            start_segment += segment_length
        segment_embed = segment_embed * example[mask_field]
        example[segment_id_field] = segment_embed
        return example

    def _calc_mask(self, example, word_id_field='word_id', mask_field='mask', **kwargs):
        word_ids = example[word_id_field]
        mask = tf.logical_not(tf.math.equal(word_ids, 0))
        example[mask_field] = tf.cast(mask, tf.int32)
        return example

    def create_tf_dataset(self, csv_filename='preprocessed_ner_dataset.csv', BATCH_SIZE=128, BUFFER_SIZE=2048,
                          test_set_batches=75, labelled_train_batches=20, **kwargs):

        tf_ds = tf.data.experimental.make_csv_dataset(os.path.join(ner.DATA_DIR, csv_filename), batch_size=self.max_seq_length,
                                                      shuffle=False, num_epochs=1)

        tf_ds_preprocessed = (
            tf_ds
                .map(self._calc_mask)
                .map(self._get_segments)
                .shuffle(BUFFER_SIZE)
                .batch(BATCH_SIZE, drop_remainder=True)
                .prefetch(1)
        )

        test_ds = tf_ds_preprocessed.take(test_set_batches)
        train_ds = tf_ds_preprocessed.skip(test_set_batches)
        labelled_ds = train_ds.take(labelled_train_batches)
        unlabelled_ds = train_ds.skip(labelled_train_batches)

        return labelled_ds, unlabelled_ds, test_ds