
#!/usr/bin/env python3
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DataAnalyzer(object):
    text_data = None
    path = None
    directory = None
    figs = []

    def __init__(self, file):
        try:
            self.text_data = pd.read_csv(file, index_col=0)
        except FileNotFoundError:
            print("Please provide a valid path to the file containing textual data!")
            return
        self.path = os.path.dirname(file)
        file_name = os.path.splitext(os.path.basename(file))[0]
        self.directory = os.path.join(self.path, "analysis", file_name)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def visualize(self):
        if self.text_data is None:
            return

        fig = plt.figure("class distribution", figsize=(10,6))
        sns.barplot(x=self.target_dist.index.astype(str), y=self.target_dist, hue=self.target_dist.index.astype(str), palette="magma")
        plt.xlabel("Product Type Code")
        plt.xticks(rotation=45)
        plt.ylabel("Proportion")
        plt.text(0, max(self.target_dist.values), "Total number of classes: {}".format(len(self.target_dist)))
        self.figs.append(fig)

        fig = plt.figure("language distribution", figsize=(10,6))
        ax1=plt.subplot(1,2,1)
        sns.barplot(x=self.lang_dist[:3].index, y=self.lang_dist[:3], hue=self.lang_dist[:3].index, palette="magma")
        plt.xlabel("Language Code")
        plt.ylabel("Proportion")
        ax2=plt.subplot(1,2,2)
        sns.barplot(x=self.lang_dist[3:].index, y=self.lang_dist[3:], hue=self.lang_dist[3:].index, palette="magma")
        plt.xlabel("Language Code")
        plt.ylabel("Proportion")
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        ax1.set_position([pos1.x0, pos1.y0, pos1.width*0.5, pos1.height])
        ax2.set_position([pos1.x1-0.1, pos2.y0, pos2.width*1.6, pos2.height])
        self.figs.append(fig)

        fig = plt.figure("text length ~ product type distribution", figsize=(10,6))
        sns.boxplot(x="prdtypecode", y="length", data=self.text_data, color="rosybrown", order=self.target_dist.index)
        plt.xticks(rotation=45)
        plt.xlabel("Product Type Code")
        plt.ylabel("Text Length")
        self.figs.append(fig)

        fig = plt.figure("text length ~ language distribution", figsize=(10,6))
        ax1=plt.subplot(2,1,1)
        sns.boxplot(x="language", y="length", data=self.text_data, color="rosybrown", order=self.lang_dist.index)
        plt.xlabel("Language Code")
        plt.ylabel("Text Length")
        ax2=plt.subplot(2,1,2)
        sns.boxplot(x="language", y="length", data=self.text_data, color="rosybrown", order=self.lang_dist.index)
        plt.ylim([0,100])
        plt.xlabel("Language Code")
        plt.ylabel("Text Length")
        self.figs.append(fig)

        plt.show()

    def analyze(self):
        if self.directory is None:
            return

        self.text_data['language'] = self.text_data['language'].str.capitalize()
        self.lang_dist = self.text_data['language'].value_counts(normalize=True).sort_values(ascending=False)
        self.target_dist = self.text_data['prdtypecode'].value_counts(normalize=True).sort_values(ascending=False)

        # analyze outliers by text length
        outlier_length = self.text_data[self.text_data['length']>2500]
        outlier_length.to_csv(os.path.join(self.directory, "text_length_outliers.csv"))

        # analyze outliers by language
        outlier_lang = self.lang_dist[self.lang_dist<0.003]
        outlier_lang = self.text_data[self.text_data['language'].isin(outlier_lang.index)]
        outlier_lang.to_csv(os.path.join(self.directory, "text_language_outliers.csv"))

    def save_figures(self):
        for fig in self.figs:
            figname = fig.get_label()+".jpg"
            fig.savefig(os.path.join(self.directory, figname))

def main():
    if len(sys.argv) < 2:
        print("Please provide the csv file of the preprocessed data")
        return 0
    path = sys.argv[1]

    vis = DataAnalyzer(path)
    vis.analyze()
    try:
        vis.visualize()
    except KeyboardInterrupt:
        print("Interrupt by keyboard")
    finally:
        vis.save_figures()

if __name__=="__main__":
    main()