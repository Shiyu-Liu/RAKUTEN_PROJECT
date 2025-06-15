#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.applications.efficientnet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

TEXT_FILE = "../data/text_data_clean.csv"
X_TRAIN_FILE = "../data/X_train.csv"
IMAGE_RATIO = "../data/x_train_with_ratios.csv"
IMAGE_DIR = "../data/images/image_train_zoomed"
IMAGE_TEST_SET = "../data/image_data_val.csv"
TEST_SET_OUTPUT = "../data/test_set_final_evaluation.csv"
OUTPUT_DIR_ORI = "results/fusion_model"
TEXT_PROB_OUTPUT = "text_pred_prob.csv"
IMG_PROB_OUTPUT = "image_pred_prob.csv"

missing_images = {'image_1142089742_product_884747735.jpg',
                  'image_1302249863_product_3793782107.jpg',
                  'image_1137819811_product_1892606336.jpg',
                  'image_1271791205_product_3894592691.jpg'}

BEST_IMAGE_MODEL = "results/efficientnetb0_b128l5_best_model.keras"
BEST_TEXT_MODEL = "results/distilbert_best_model/saved_model"

WEIGHTS = [2, 2]  # weights of [textual, image] model
OUTPUT_DIR = "results/fusion_model_"+str(WEIGHTS[0])+"_"+str(WEIGHTS[1])

def prepare_test_set_new():
    text_data = pd.read_csv(TEXT_FILE, delimiter=';', index_col=0)
    X_train = pd.read_csv(X_TRAIN_FILE, index_col=0)
    ratio = pd.read_csv(IMAGE_RATIO)
    X_train['imagefile'] = "image_"+X_train['imageid'].astype(str)+"_product_"+X_train['productid'].astype(str)+".jpg"
    X_train['ratio'] = ratio['content_ratio']

    max_index = X_train.index[-1]
    text_data_no_aug = text_data.loc[:max_index]
    _, X_test, _, y_test = train_test_split(text_data_no_aug['text'], text_data_no_aug['prdtypecode'], test_size=0.2, random_state=42, stratify=text_data_no_aug['prdtypecode'])

    test_ratio = X_train.loc[X_test.index, 'ratio']
    X_test.drop(test_ratio[test_ratio<0.04].index, inplace=True)
    y_test.drop(test_ratio[test_ratio<0.04].index, inplace=True)
    print("Test set Distribution:\n", y_test.value_counts())

    test_dataset = pd.concat([X_test, X_train.loc[X_test.index, 'imagefile'], y_test.loc[X_test.index]], axis=1)
    missing_file = test_dataset['imagefile'].apply(lambda x: x in missing_images)
    if any(missing_file.astype(bool)==True):
        image_file = test_dataset.loc[missing_file.astype(bool)==True, 'imagefile']
        print(f"Following image files are missing:\n{image_file}")
        return False
    print(test_dataset.head())
    test_dataset = test_dataset.sort_index(ascending=True)
    test_dataset.to_csv(TEST_SET_OUTPUT, sep=';')
    return True

def prepare_test_set():
    text_data = pd.read_csv(TEXT_FILE, delimiter=';', index_col=0)
    image_test_set = pd.read_csv(IMAGE_TEST_SET, delimiter=';', index_col=0)
    print(image_test_set.head())
    image_test_set['imagefile'] = "image_"+image_test_set['imageid'].astype(str)+"_product_"+image_test_set['productid'].astype(str)+".jpg"

    _, text_test_set, _, text_y_test = train_test_split(text_data['text'], text_data['prdtypecode'], test_size=0.3, random_state=27, stratify=text_data['prdtypecode'])
    text_test_set = text_test_set.sort_index(ascending=True)

    common_idx = text_test_set.index.intersection(image_test_set.index)
    image_test_common = image_test_set.loc[common_idx].sort_index(ascending=True)
    text_test_common = text_test_set.loc[common_idx].sort_index(ascending=True)
    y_test_common = text_y_test.loc[common_idx].sort_index(ascending=True)

    print("Test set distribution:\n", text_data.loc[common_idx, 'prdtypecode'].value_counts())
    test_dataset = pd.concat([text_test_common, image_test_common['imagefile'], y_test_common], axis=1)
    missing_file = test_dataset['imagefile'].apply(lambda x: x in missing_images)
    if any(missing_file.astype(bool)==True):
        image_file = test_dataset.loc[missing_file.astype(bool)==True, 'imagefile']
        print(f"Following image files are missing:\n{image_file}")
        return False
    print(test_dataset.head())
    test_dataset = test_dataset.sort_index(ascending=True)
    test_dataset.to_csv(TEST_SET_OUTPUT, sep=';')
    return True

def generate_text_outputs(X_test, labels):
    num_labels = len(labels)
    text_model = AutoModelForSequenceClassification.from_pretrained(BEST_TEXT_MODEL, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(BEST_TEXT_MODEL)
    text_model.eval()
    text_outputs = np.zeros((X_test.shape[0], num_labels))
    for i, text in enumerate(X_test['text']):
        print(f"Predicting by text model: row {i+1}/{X_test.shape[0]}")
        # tokenize inputs
        input = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=128)
        # make preductions
        text_output = text_model(**input)
        text_prob = torch.nn.functional.softmax(text_output.logits)
        text_outputs[i,:] = text_prob.detach().numpy()
    text_outputs = pd.DataFrame(data=text_outputs, index=X_test.index, columns=[f'class_{i}' for i in labels])
    return text_outputs

def generate_image_outputs(X_test, labels):
    img_model = load_model(BEST_IMAGE_MODEL)
    img_outputs = np.zeros((X_test.shape[0], len(labels)))
    for i, file in enumerate(X_test['imagefile']):
        print(f"Predicting by image model: row {i+1}/{X_test.shape[0]}")
        # load image
        image_path = os.path.join(IMAGE_DIR, file)
        img = load_img(image_path, target_size=(224,224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # make predictions
        preds = img_model.predict(x)
        img_outputs[i,:] = preds
    img_outputs = pd.DataFrame(data=img_outputs, index=X_test.index, columns=[f'class_{i}' for i in labels])
    return img_outputs

def main():
    if not os.path.exists(TEST_SET_OUTPUT):
        if not prepare_test_set():
            print("Evaluation aborted!")
            return 0

    test_set = pd.read_csv(TEST_SET_OUTPUT, delimiter=';', index_col=0)
    X_test = test_set.drop(['prdtypecode'], axis=1)
    y_test = test_set['prdtypecode']
    labels = y_test.value_counts().sort_index(ascending=True).index

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # predict by text model
    text_prob_file = os.path.join(OUTPUT_DIR, TEXT_PROB_OUTPUT)
    if not os.path.exists(text_prob_file):
        text_prob_file_ori = os.path.join(OUTPUT_DIR_ORI, TEXT_PROB_OUTPUT)
        if not os.path.exists(text_prob_file_ori):
            text_outputs = generate_text_outputs(X_test, labels)
            text_outputs.to_csv(text_prob_file)
            text_outputs.to_csv(text_prob_file_ori)
        else:
            text_outputs = pd.read_csv(text_prob_file_ori, index_col=0, header=0)
            text_outputs.to_csv(text_prob_file)
    else:
        text_outputs = pd.read_csv(text_prob_file, index_col=0, header=0)

    # image model prediction
    image_prob_file = os.path.join(OUTPUT_DIR, IMG_PROB_OUTPUT)
    if not os.path.exists(image_prob_file):
        image_prob_file_ori = os.path.join(OUTPUT_DIR_ORI, IMG_PROB_OUTPUT)
        if not os.path.exists(image_prob_file_ori):
            img_outputs = generate_image_outputs(X_test, labels)
            img_outputs.to_csv(image_prob_file)
            img_outputs.to_csv(image_prob_file_ori)
        else:
            img_outputs = pd.read_csv(image_prob_file_ori, index_col=0, header=0)
            img_outputs.to_csv(image_prob_file)
    else:
        img_outputs = pd.read_csv(image_prob_file, index_col=0, header=0)

    # combine results of two models to predict the final class
    pred_prob = np.zeros((text_outputs.shape[0], len(labels)))
    y_pred = np.zeros((text_outputs.shape[0], 1))
    y_pred_text = np.zeros((text_outputs.shape[0], 1))
    y_pred_img = np.zeros((text_outputs.shape[0], 1))
    total_weigths = sum(WEIGHTS)
    i = 0
    for (_, text_prob), (_, img_prob) in zip(text_outputs.iterrows(), img_outputs.iterrows()):
        prob = (text_prob.to_numpy()*WEIGHTS[0]+img_prob.to_numpy()*WEIGHTS[1])/total_weigths
        pred_prob[i,:] = prob
        idx = np.argmax(prob)
        y_pred[i] = labels[idx]
        idx_text = np.argmax(text_prob.to_numpy())
        y_pred_text[i] = labels[idx_text]
        idx_img = np.argmax(img_prob.to_numpy())
        y_pred_img[i] = labels[idx_img]
        i+=1
    y_pred = pd.DataFrame(data=y_pred, index=test_set.index, columns=['prdtypecode']).astype(int)
    y_pred_text = pd.DataFrame(data=y_pred_text, index=test_set.index, columns=['prdtypecode']).astype(int)
    y_pred_img = pd.DataFrame(data=y_pred_img, index=test_set.index, columns=['prdtypecode']).astype(int)
    pred_prob = pd.DataFrame(data=pred_prob, index=test_set.index, columns=[f'class_{i}' for i in labels])

    # save to file
    pred_prob.to_csv(os.path.join(OUTPUT_DIR, 'y_pred_prob.csv'))
    y_pred.to_csv(os.path.join(OUTPUT_DIR, 'y_pred.csv'))
    y_pred_text.to_csv(os.path.join(OUTPUT_DIR, 'y_pred_text.csv'))
    y_pred_img.to_csv(os.path.join(OUTPUT_DIR, 'y_pred_img.csv'))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels, normalize='true')
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Proportion'})

    plt.xlabel('Predicted Class', fontsize=14)
    plt.ylabel('Real Class', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    df_cm = pd.DataFrame(cm, index=[f"Actual {label}" for label in labels],
                        columns=[f"Predicted {label}" for label in labels])
    print(df_cm)
    # classification report
    df_report_text = pd.DataFrame(classification_report(y_test, y_pred_text, output_dict=True)).transpose()
    print("Text model:\n", df_report_text.round(2))
    df_report_img = pd.DataFrame(classification_report(y_test, y_pred_img, output_dict=True)).transpose()
    print("Image model:\n", df_report_img.round(2))
    df_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    print("Fusion model:\n", df_report.round(2))

    # save to file
    df_cm.to_csv(os.path.join(OUTPUT_DIR, 'confusion_matrix.csv'))
    df_report.to_csv(os.path.join(OUTPUT_DIR, 'classification_report.csv'))
    df_report_text.to_csv(os.path.join(OUTPUT_DIR, 'classification_report_text_model.csv'))
    df_report_img.to_csv(os.path.join(OUTPUT_DIR, 'classification_report_img_model.csv'))

if __name__=='__main__':
    main()