from ner.trainers.baseline_ner_trainer import NER_BaselineTraininer
from ner.models.baseline_ner import BaselineNER
from ner.preprocessor import Preprocessor

p = Preprocessor()

labelled_ds, unlabelled_ds, test_ds = p.create_tf_dataset()

trainer = NER_BaselineTraininer()


model = trainer.train(labelled_ds=labelled_ds, test_ds=test_ds)