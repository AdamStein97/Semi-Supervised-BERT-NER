import os
import ner
from ner.preprocessor import Preprocessor
from ner.utils import load_config
from ner.models.bert_ner import BERT_NER
from ner.model_evaluator import ModelEvaluator

config = load_config(model_config_name='bert_ner_confidence_kl_config.yaml', master_config_name='config.yaml')

p = Preprocessor(**config)

labelled_ds, unlabelled_ds, test_ds = p.create_tf_dataset(**config)

model = BERT_NER(**config)

model.load_weights(os.path.join(ner.MODEL_DIR, 'BERT_NER_confidence_kl_final'))

model_evaluator = ModelEvaluator(test_ds,  results_name='BERT_confidence_kl')

# model_evaluator.plot_all_latent_images(model, **config)

y_true_numpy_test_set, y_pred_numpy_test_set = model_evaluator.get_test_ds_labels(model)

model_evaluator.get_all_results(y_true_numpy_test_set, y_pred_numpy_test_set)



