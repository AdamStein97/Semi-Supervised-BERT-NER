import os
import ner
from ner.preprocessor import Preprocessor
from ner.utils import load_config
from ner.models.baseline_ner import BaselineNER
from ner.model_evaluator import ModelEvaluator

config = load_config(model_config_name='baseline_ner_config.yaml', master_config_name='config.yaml')

p = Preprocessor(**config)

labelled_ds, unlabelled_ds, test_ds = p.create_tf_dataset(**config)

model = BaselineNER(**config)

model.load_weights(os.path.join(ner.MODEL_DIR, 'NER_baseline_final'))

model_evaluator = ModelEvaluator(test_ds,  results_name='NER_baseline')

model_evaluator.plot_all_latent_images(model, **config)

# y_true_numpy_test_set, y_pred_numpy_test_set = model_evaluator.get_test_ds_labels(model)
#
# model_evaluator.get_all_results(y_true_numpy_test_set, y_pred_numpy_test_set)
#


