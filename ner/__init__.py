import tensorflow as tf
import os

tf.random.set_seed(99)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
MODEL_DIR = os.path.join(ROOT_DIR, "saved_models")
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")