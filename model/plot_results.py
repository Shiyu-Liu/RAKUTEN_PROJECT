import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fusion_model_dir = "fusion_model_2_2"
text_results = "results/"+fusion_model_dir+"/classification_report_text_model.csv"
image_results = "results/"+fusion_model_dir+"/classification_report_img_model.csv"
fusion_results = "results/"+fusion_model_dir+"/classification_report.csv"

def plot_f1_score_comparison():
    df_text = pd.read_csv(text_results, index_col=0)
    df_image = pd.read_csv(image_results, index_col=0)
    df_fusion = pd.read_csv(fusion_results, index_col=0)

    df_text = df_text.iloc[:27].rename({'f1-score': 'Text'}, axis=1)
    df_image = df_image.iloc[:27].rename({'f1-score': 'Image'}, axis=1)
    df_fusion = df_fusion.iloc[:27].rename({'f1-score': 'Fusion'}, axis=1)

    df_all = pd.concat([df_image['Image'], df_text['Text'], df_fusion['Fusion']], axis=1)
    df_all = df_all.reset_index().rename(columns={'index': 'prdtypecode'})

    df_plot = df_all.melt(
        id_vars='prdtypecode',
        value_vars=['Image', 'Text', 'Fusion'],
        var_name='model',
        value_name='f1_score'
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x='prdtypecode', y='f1_score', hue='model', palette='Set2')
    plt.xlabel('Product Type Code', fontsize=14)
    plt.ylabel('F1-Score', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Model', fontsize=12, title_fontsize=13, bbox_to_anchor=(1.13, 1), loc='upper right')
    plt.tight_layout()
    plt.show()

def main():
    plot_f1_score_comparison()

if __name__=="__main__":
    main()