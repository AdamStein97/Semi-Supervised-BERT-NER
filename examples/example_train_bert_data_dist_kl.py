from ner.trainers.bert_ner_trainer_data_dist_kl import BERT_NER_TraininerDataDistKL
from ner.preprocessor import Preprocessor

p = Preprocessor()

labelled_ds, unlabelled_ds, test_ds = p.create_tf_dataset()

trainer = BERT_NER_TraininerDataDistKL()

model = trainer.train(labelled_ds=labelled_ds, test_ds=test_ds, unlabelled_ds=unlabelled_ds)