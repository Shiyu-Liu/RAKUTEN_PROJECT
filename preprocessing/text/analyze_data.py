
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
        ax1 = plt.gca()
        sns.barplot(x=self.target_dist_abs.index.astype(str), y=self.target_dist_abs, hue=self.target_dist_abs.index.astype(str), palette="magma")
        ax1.set_xlabel("Product Type Code")
        ax1.set_ylabel("Total Number")
        ax1.tick_params(axis='x', labelrotation=45)
        ax2 = ax1.twinx()
        sns.barplot(x=self.target_dist.index.astype(str), y=self.target_dist, hue=self.target_dist.index.astype(str), palette="magma")
        ax2.set_ylabel("Proportion")
        plt.text(0, max(self.target_dist.values), "Total number of classes: {}".format(len(self.target_dist)))
        plt.xticks(rotation=45)
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

        fig = plt.figure("class ~ language distribution", figsize=(10,6))
        x1 = self.text_data.loc[self.text_data['language']==self.lang_dist.index.astype(str)[0], 'prdtypecode'].value_counts().reindex(self.target_dist.index)
        plt.bar(x1.index.astype(str), x1, label=self.lang_dist.index.astype(str)[0])
        x2 = self.text_data.loc[self.text_data['language']==self.lang_dist.index.astype(str)[1], 'prdtypecode'].value_counts().reindex(self.target_dist.index)
        plt.bar(x2.index.astype(str), x2, label=self.lang_dist.index.astype(str)[1], bottom=x1)
        x3 = self.text_data.loc[self.text_data['language']==self.lang_dist.index.astype(str)[2], 'prdtypecode'].value_counts().reindex(self.target_dist.index)
        plt.bar(x3.index.astype(str), x3, label=self.lang_dist.index.astype(str)[2], bottom=x1+x2)
        x4 = self.text_data.loc[self.text_data['language'].isin(self.lang_dist.index.astype(str)[3:]), 'prdtypecode'].value_counts().reindex(self.target_dist.index)
        plt.bar(x4.index.astype(str), x4, label='Others', bottom=x1+x2+x3, color="grey")
        plt.xlabel("Language Code")
        plt.ylabel("Total Number")
        plt.xticks(rotation=45)
        plt.legend()
        self.figs.append(fig)

        fig = plt.figure("text length (characters) ~ product type distribution", figsize=(10,6))
        sns.boxplot(x="prdtypecode", y="c_length", data=self.text_data, color="rosybrown", order=self.target_dist.index)
        plt.xticks(rotation=45)
        plt.xlabel("Product Type Code")
        plt.ylabel("Number of Characters")
        self.figs.append(fig)

        fig = plt.figure("text length (words) ~ product type distribution", figsize=(10,6))
        sns.boxplot(x="prdtypecode", y="w_length", data=self.text_data, color="rosybrown", order=self.target_dist.index)
        plt.xticks(rotation=45)
        plt.xlabel("Product Type Code")
        plt.ylabel("Number of Words")
        self.figs.append(fig)

        fig = plt.figure("text length (characters) ~ language distribution", figsize=(10,6))
        ax1=plt.subplot(2,1,1)
        sns.boxplot(x="language", y="c_length", data=self.text_data, color="rosybrown", order=self.lang_dist.index)
        plt.xlabel("Language Code")
        plt.ylabel("Number of Characters")
        ax2=plt.subplot(2,1,2)
        sns.boxplot(x="language", y="c_length", data=self.text_data, color="rosybrown", order=self.lang_dist.index)
        plt.ylim([0,100])
        plt.xlabel("Language Code")
        plt.ylabel("Number of Characters")
        self.figs.append(fig)

        fig = plt.figure("text length (characters) distribution", figsize=(10,6))
        ax1=plt.subplot(1,2,1)
        sns.boxplot(x="c_length", data=self.text_data)
        plt.xlabel("Number of Characters")
        ax2=plt.subplot(1,2,2)
        sns.boxplot(x="c_length", data=self.text_data)
        plt.xlabel("Number of Characters")
        plt.xlim([0,200])
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        ax1.set_position([pos1.x0, pos1.y0, pos1.width*1.6, pos1.height])
        ax2.set_position([pos1.x0+pos1.width*1.6+0.05, pos2.y0, pos2.width*0.5, pos2.height])
        self.figs.append(fig)

        fig = plt.figure("text length (words) distribution", figsize=(10,6))
        ax1=plt.subplot(1,2,1)
        sns.boxplot(x="w_length", data=self.text_data)
        plt.xlabel("Number of Words")
        ax2=plt.subplot(1,2,2)
        sns.boxplot(x="w_length", data=self.text_data)
        plt.xlabel("Number of Words")
        plt.xlim([0,50])
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        ax1.set_position([pos1.x0, pos1.y0, pos1.width*1.6, pos1.height])
        ax2.set_position([pos1.x0+pos1.width*1.6+0.05, pos2.y0, pos2.width*0.5, pos2.height])
        self.figs.append(fig)
        plt.show()

    def analyze(self):
        if self.directory is None:
            return

        self.text_data['language'] = self.text_data['language'].str.capitalize()
        self.lang_dist = self.text_data['language'].value_counts(normalize=True).sort_values(ascending=False)
        self.target_dist = self.text_data['prdtypecode'].value_counts(normalize=True).sort_values(ascending=False)
        self.target_dist_abs = self.text_data['prdtypecode'].value_counts(normalize=False).sort_values(ascending=False)

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