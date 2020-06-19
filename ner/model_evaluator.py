import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import os
import ner


class ModelEvaluator():
    def __init__(self, test_ds, id2tag=None, label_distribution=None, results_name='model'):
        self.test_ds = test_ds
        self.results_name = results_name
        if id2tag is None:
            id2tag = {
              1: 'O',
              2: 'geo',
              3: 'gpe',
              4: 'per',
              5: 'org',
              6: 'tim',
              7: 'art',
              8: 'nat',
              9: 'eve'}

        if label_distribution is None:
            label_distribution = np.array([8.468e-01, 4.300e-02, 1.530e-02, 3.270e-02, 3.520e-02, 2.560e-02,
                                         7.000e-04, 2.000e-04, 5.000e-04])

        self.id2tag = id2tag
        self.label_distribution = label_distribution
        self.ids = list(id2tag.keys())


    def get_confusion_matricies_from_numpy(self, y_true_flat, y_pred_flat):
        pred_norm_cm = confusion_matrix(y_true_flat, y_pred_flat, labels=self.ids, normalize='pred')
        true_norm_cm = confusion_matrix(y_true_flat, y_pred_flat, labels=self.ids, normalize='true')
        return pred_norm_cm, true_norm_cm

    def plot_confusion(self, matrix, matrix_name='confusion_matrix'):
        matrix = np.round(matrix, 2)

        mat_df = pd.DataFrame()
        for i, id_true in enumerate(self.ids):
            for j, id_pred in enumerate(self.ids):
                mat_df = mat_df.append(pd.DataFrame({'True Category': self.id2tag[id_true], 'Predicted Category': self.id2tag[id_pred], 'val':[matrix[i][j]]}), ignore_index=True)

        mat_df = mat_df.pivot('True Category', 'Predicted Category', 'val')

        fig = plt.figure(figsize=(10,10))
        plt.tight_layout()
        ax = sns.heatmap(mat_df, annot=True, linewidths=.5, cmap="YlGnBu")
        b, t = plt.ylim() # discover the values for bottom and top
        b += 0.5 # Add 0.5 to the bottom
        t -= 0.5 # Subtract 0.5 from the top
        plt.ylim(b, t) # update the ylim(bottom, top) values
        plt.title(matrix_name)
        plt.savefig(os.path.join(ner.RESULTS_DIR, self.results_name + matrix_name + ".png"))
        plt.show()

    def get_results_df(self, pred_norm_cm, true_norm_cm, y_pred_flat=None):
        prec = np.diagonal(true_norm_cm)
        recall = np.diagonal(pred_norm_cm)
        results_df = pd.DataFrame({"Precision": prec, "Recall": recall})
        results_df['F1'] = results_df.apply(lambda x: (2 * x["Precision"] * x["Recall"]) / (x["Precision"] + x["Recall"]), axis=1).fillna(0)
        results_df['Catgegory'] = list(self.id2tag.values())
        results_df['True Label Distribution'] = np.round(self.label_distribution, 4)
        if y_pred_flat is not None:
            pred_labels_occurances = [ np.round(y_pred_flat.tolist().count(i) / len(y_pred_flat), 4) for i in range(1,10)]
            results_df['Pred Label Distribution'] = pred_labels_occurances

        return round(results_df, 3)

    def get_test_ds_labels(self, model):
        y_true_numpy = None
        y_pred_numpy = None
        for i, x in enumerate(self.test_ds):
            label = x['tag_id']
            _, pred_y = model(x['word_id'], x['segment_id'], x['mask'])
            pred_labels = tf.argmax(pred_y, axis=-1)
            if i == 0:
              y_true_numpy = label.numpy()
              y_pred_numpy = pred_labels.numpy()
            else:
              y_true_numpy = np.concatenate((y_true_numpy, label.numpy()), axis=0)
              y_pred_numpy = np.concatenate((y_pred_numpy, pred_labels.numpy()), axis=0)
        return y_true_numpy, y_pred_numpy

    def get_all_results(self, y_true_numpy, y_pred_numpy):
        y_true_flat = y_true_numpy.flatten()
        y_pred_flat = y_pred_numpy.flatten()
        label_filter_0 = y_true_flat != 0
        y_true_flat = y_true_flat[label_filter_0]
        y_pred_flat = y_pred_flat[label_filter_0]
        pred_norm_cm, true_norm_cm = self.get_confusion_matricies_from_numpy(y_true_flat, y_pred_flat)
        self.plot_confusion(pred_norm_cm, "Pred_Norm_CM")
        self.plot_confusion(true_norm_cm, "True_Norm_CM")

        print("Category Results\n")
        results_df = self.get_results_df(pred_norm_cm, true_norm_cm, y_pred_flat)
        print(results_df)
        results_df.to_csv(os.path.join(ner.RESULTS_DIR, self.results_name + "_results.csv"))

        print("\nResults Describe\n")
        describe_df = results_df.describe()[["Precision",  "Recall", "F1"]][1:]
        print(describe_df)
        describe_df.to_csv(os.path.join(ner.RESULTS_DIR, self.results_name + "_describe_results.csv"))

        print("\n")
        acc = 100 * np.sum(y_true_flat == y_pred_flat) / len(y_true_flat)
        print("Accuracy: {:.2f}%".format(acc))

        label_filter_1 = y_true_flat != 1
        acc_no_o = 100 * np.sum(y_true_flat[label_filter_1] == y_pred_flat[label_filter_1]) / len(y_true_flat[label_filter_1])
        print("Accuracy No O: {:.2f}%".format(acc_no_o))

        print("Mean F1: {:.4f}".format(results_df['F1'].mean()))
