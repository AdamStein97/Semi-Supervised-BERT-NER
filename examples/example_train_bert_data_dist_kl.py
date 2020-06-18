from ner.trainers.bert_ner_trainer_data_dist_kl import BERT_NER_TraininerDataDistKL
from ner.preprocessor import Preprocessor
from ner.utils import load_config

config = load_config(model_config_name='bert_ner_data_dist_kl_config.yaml', master_config_name='config.yaml')


p = Preprocessor(**config)

labelled_ds, unlabelled_ds, test_ds = p.create_tf_dataset(**config)

trainer = BERT_NER_TraininerDataDistKL(**config)

model = trainer.train(labelled_ds=labelled_ds, test_ds=test_ds, unlabelled_ds=unlabelled_ds, **config)