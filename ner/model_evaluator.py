import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
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

    def get_2d_encodings(self, model, pca=None, take=10, filter_O=True, BATCH_SIZE=128, max_seq_length=50, **kwargs):
        all_encoded_words = []
        all_encoded_labels = []
        all_pred_labels = []
        for x in self.test_ds.take(take):
            label = x['tag_id']
            encoding, pred_labels = model(input_word_ids=x['word_id'], segment_ids=x['segment_id'], input_mask=x['mask'])
            word_encoding = tf.reshape(encoding, [BATCH_SIZE * max_seq_length, -1])
            word_labels = tf.reshape(label, [BATCH_SIZE * max_seq_length, -1])
            pred_labels = tf.argmax(tf.reshape(pred_labels, [BATCH_SIZE * max_seq_length, -1]), axis=-1)
            all_encoded_words.append(word_encoding.numpy())
            all_encoded_labels.append(word_labels.numpy())
            all_pred_labels.append(pred_labels.numpy())

        words = np.concatenate(tuple(all_encoded_words))
        labels = np.concatenate(tuple(all_encoded_labels))
        pred_labels = np.concatenate(tuple(all_pred_labels))

        if filter_O:
            label_fil = np.squeeze([label != 0 and label != 1 for label in labels])
        else:
            label_fil = np.squeeze([label != 0 for label in labels])

        words = words[label_fil]
        labels = labels[label_fil]
        pred_labels = pred_labels[label_fil]

        if pca is None:
            pca = PCA(n_components=2)
            pca.fit(words)

        reduced_x = pca.transform(words)
        return pca, reduced_x, labels, pred_labels #labels_tag, pred_labels_tag

    def plot_latent(self, x_2d, labels, save_name="latent_space.png"):
        fig, ax = plt.subplots(figsize=(10, 10))

        scatter = ax.scatter(x_2d[:, 0], x_2d[:, 1], c=np.squeeze(labels))

        # # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="lower left", title="Classes")
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)

        legend_items = np.sort(np.unique(labels))

        for i in range(len(legend1.get_texts())):
            legend1.get_texts()[i].set_text(self.id2tag[legend_items[i]])


        plt.savefig(os.path.join(ner.RESULTS_DIR, save_name))
        plt.show()

    def plot_all_latent_images(self, model, **kwargs):
        _, x_2d, labels, pred_labels = self.get_2d_encodings(model, filter_O=False, **kwargs)
        self.plot_latent(x_2d, pred_labels, save_name=self.results_name + "_predicted_labels_2d.png")
        self.plot_latent(x_2d, labels, save_name=self.results_name + "_true_labels_2d.png")

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
            _, pred_y = model(input_word_ids=x['word_id'], segment_ids=x['segment_id'], input_mask=x['mask'])
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
