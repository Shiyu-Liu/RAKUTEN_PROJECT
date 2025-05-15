#!/usr/bin/env python3
import os
import sys
import re
import pandas as pd
import numpy as np
from langdetect import detect
import html
import tiktoken

XTrainFile = "X_train.csv"
YTrainFile = "Y_train.csv"
XTestFile = "X_test.csv"
SaveFileSuffix = "clean"
SaveDataFilename = "text_data_clean"
SaveDirName = "preprocess"

Filtering_Params={'min_words_length': 3,
                  'max_words_length': 100,
                  'target_sample_size': 2500}

class TextPreProcessing(object):
    X_train = None
    y_train = None
    X_test = None
    text_data = None

    def __init__(self, path=None):
        if path == None:
            print("Please provide relative or global path to data directory")
            return
        self.data_path = path
        self.save_path = os.path.join(path, SaveDirName)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def read_text(self):
        try:
            self.X_train = pd.read_csv(os.path.join(self.data_path, XTrainFile), index_col=0)
            self.y_train = pd.read_csv(os.path.join(self.data_path, YTrainFile), index_col=0)
            self.X_test = pd.read_csv(os.path.join(self.data_path, XTestFile), index_col=0)
        except FileNotFoundError as e:
            print("Please check the path provided, error message:{}".format(e))
            return False
        return True

    def detect_language(self, column: pd.Series):
        detected_lang = []
        rows = column.shape[0]
        for i, text in enumerate(column):
            lang = ""
            if text is not None:
                lang = detect(text)
            detected_lang.append(lang)
            print("Row {}/{}, detected language: {}".format(i+1, rows, lang))
        return pd.Series(data=detected_lang, index=column.index)

    def clean_text(self, X: pd.Series):
        texts = []
        rows = X.shape[0]
        for i, text in enumerate(X):
            # remove html tags
            text = re.sub(r'<[^>]*>', '', text)
            # interprete html encodings
            text = html.unescape(text)
            # remove units patterns like "xx cm", "xx kg" or "xx kg/m" etc.
            text = re.sub(r'\b\d+(?:\.\d+)?\s?(?:[a-zA-Z%]{1,3}(?:[/.][a-zA-Z]{1,2})?)\b', ' ', text)
            # remove units patterns of "Axx" or "N°"
            text = re.sub(r'\bA\d+\b|\bN°?\s\d+\b', '', text)
            # remove unwanted special characters (keeping symbolic letters and whitespace)
            text = re.sub(r'[^a-zA-Z\s\'À-ÿäöüÄÖÜß]', ' ', text, flags=re.UNICODE)
            # remove dimension pattern with symbol 'Ø'
            text = re.sub(r'\s[Ø]\s', ' ', text, flags=re.UNICODE)
            # remove remaining dimension patterns like " X " or " x "
            text = re.sub(r'\s[×xX]\s', ' ', text)
            # remove multiple whitespace characters and the leading/trailing whitespace
            text = re.sub(r'\s+', ' ', text).lstrip().rstrip()
            print("Cleaning row {}/{}".format(i+1, rows))
            texts.append(text)
        return pd.Series(data=texts, index=X.index)

    def preprocess_from_raw(self, X: pd.DataFrame):
        # merge 'designation' and 'description' columns
        X['description'] = X['description'].fillna("")
        X['text'] = X['designation'] + " " + X['description']
        # clean up special characters
        X['text'] = self.clean_text(X['text'])
        # construst the image filename corresponding to the format image_<image_id>_product_<product_id>.jpg
        X['imagefile'] = "image_" + X['imageid'].astype(str) + "_product_" + X['productid'].astype(str) + ".jpg"
        # drop unnecessary columns
        X.drop(['description', 'designation', 'imageid', 'productid'], axis=1, inplace=True)
        # rename 'text' column to 'description'
        X.rename({'text':'description'}, axis=1, inplace=True)
        # compute the text length (number of characters) for each product's description
        X['c_length'] = X['description'].apply(len)
        # compute the number of words for each product's description
        X['w_length'] = X['description'].apply(lambda x: len(re.findall(r"\b\w+(?:'\w+)?\b", x)))
        # detect language and save to a new column
        X['language'] = self.detect_language(X['description'])
        return X

    def preprocess_step1(self, save_csv=False):
        if self.data_path is None or not self.read_text():
            return

        # process the raw data
        self.X_train = self.preprocess_from_raw(self.X_train)
        self.X_test = self.preprocess_from_raw(self.X_test)
        if save_csv:
            self.X_test.to_csv(os.path.join(self.data_path, os.path.splitext(XTestFile)[0]+'_'+SaveFileSuffix+'.csv'))

        # concatenate train and target data (X, y) into a single variable
        self.text_data = self.X_train.join(self.y_train)
        if save_csv:
            self.text_data.to_csv(os.path.join(self.save_path, SaveDataFilename+'_step1.csv'))

        self.print_info()

    def filter_text_length(self, X: pd.DataFrame, upper: int, lower: int, target_size: int, drop: str=None):
        drop_class = ""
        if drop is not None:
            drop_class = drop

        # drop entries if text length is above threshold and belong to majority class (whose size is way larger than target size)
        filtered = pd.DataFrame()
        for c in drop_class:
            filtered_upper = X[(X['w_length']>upper) & (X['prdtypecode'].astype(str) == c)]
            if len(X[X['prdtypecode'].astype(str) == c]) - len(filtered) > target_size:
                X.drop(filtered_upper.index, inplace=True)
                filtered = pd.concat([filtered, filtered_upper], axis=0)

        # drop entries if text length is lower than a threshold
        filtered_lower = X[X['w_length']<lower]
        if filtered_lower.shape[0] != 0:
            X.drop(filtered_lower.index, inplace=True)

        # save filtered entries to output
        filtered = pd.concat([filtered, filtered_lower], axis=0)

        # cap the text around the threshold and find the last acceptable punctuation within the limit
        i = 1
        rows = X.shape[0]
        for des, length in zip(X['description'], X['w_length']):
            if length > upper:
                cut_des = ' '.join(re.findall(r"\b\w+(?:'\w+)?\b", des)[:upper])
                X.loc[X['description']==des, 'w_length'] = upper
                X.loc[X['description']==des, 'c_length'] = len(cut_des)
                X.loc[X['description']==des, 'description'] = cut_des
            print("Filtering row {}/{}".format(i, rows))
            i+=1
        return X, filtered

    def preprocess_step2(self, csv=None, save_csv=False):
        if csv is None and self.text_data is None:
            print("Performing the preprocessing step 1 first.")
            self.preprocess_step1()

        if csv is not None:
            try:
                self.text_data = pd.read_csv(os.path.join(self.save_path, csv), index_col=0)
            except FileNotFoundError as e:
                print("Please check the file provided, error message:{}".format(e))
                return

        # extract filtering parameters
        upper_thres = Filtering_Params['max_words_length']
        lower_thres = Filtering_Params['min_words_length']
        target_size = Filtering_Params['target_sample_size']

        # retrieve majority classes (of which total size of samples needs to be cut down)
        prdtype_counts = self.text_data['prdtypecode'].value_counts()
        majority_classes = prdtype_counts[prdtype_counts>target_size].index.astype(str)
        self.text_data, filtered_data = self.filter_text_length(self.text_data, upper_thres, lower_thres, target_size, majority_classes)

        # sort index for filtered data
        filtered_data.sort_index(inplace=True)

        if save_csv:
            self.text_data.to_csv(os.path.join(self.save_path, SaveDataFilename+'_step2.csv'))
            filtered_data.to_csv(os.path.join(self.save_path, SaveDataFilename+'_step2_filtered.csv'))

    def downsample_majority_class(self, X: pd.DataFrame, drop_classes: list, threshold: int):
        dropped_data = pd.DataFrame()
        for c in drop_classes:
            drop_num = len(X[X['prdtypecode'].astype(str) == c]) - threshold
            print("class {}, drop_num: {}".format(c, drop_num))
            candidate = X[(X['prdtypecode'].astype(str) == c) & (X['language'].str.upper() != Target_Language)]
            if len(candidate) < drop_num:
                print("Warning: remaining data to be dropped is not enough, please redefine the dropping rule!")
                return X, dropped_data
            dropped = candidate.sample(n=drop_num, random_state=27)
            X.drop(dropped.index, inplace=True)
            dropped_data = pd.concat([dropped_data, dropped], axis=0)
        return X, dropped_data

    def preprocess_step3(self, csv=None, save_csv=False, split: int=0):
        if csv is None and self.text_data is None:
            print("Performing the preprocessing step 1 and 2 first.")
            self.preprocess_step1()
            self.preprocess_step2()

        if csv is not None:
            try:
                self.text_data = pd.read_csv(os.path.join(self.save_path, csv), index_col=0)
            except FileNotFoundError as e:
                print("Please check the file provided, error message:{}".format(e))
                return

        # retrieve majority classes (of which total size of samples needs to be cut down)
        target_size = Filtering_Params['target_sample_size']
        prdtype_counts = self.text_data['prdtypecode'].value_counts()
        majority_classes = prdtype_counts[prdtype_counts>target_size].index.astype(str)

        # drop some samples in the majority classes
        self.text_data, filtered_data = self.downsample_majority_class(self.text_data, majority_classes, target_size)

        # sort index for filtered data
        filtered_data.sort_index(inplace=True)

        # count the number of tokens (to estimate the costs before using openai api)
        enc = tiktoken.get_encoding("cl100k_base")
        self.text_data['tokens'] = self.text_data['description'].apply(lambda x: len(enc.encode(x)))
        print("Translating {} rows of data, requiring {} tokens".format(self.text_data.shape[0], self.text_data['tokens'].sum()))

        if save_csv:
            if split==0:
                self.text_data.to_csv(os.path.join(self.save_path, SaveDataFilename+'_step3.csv'))
            else: # split the data into multiple sub-parts and save to csv files individually
                if split > 0:
                    dfs = np.array_split(self.text_data, split)
                    for i, chunk in enumerate(dfs):
                        chunk.to_csv(os.path.join(self.save_path, SaveDataFilename+'_split_'+str(i+1)+'.csv'))
                else:
                    print("Don't split the data as the split number is wrong ({})".format(split))
            filtered_data.to_csv(os.path.join(self.save_path, SaveDataFilename+'_step3_filtered.csv'))

    def save_csv(self):
        self.text_data.to_csv(os.path.join(self.save_path, SaveDataFilename+'.csv'))

    def print_info(self):
        if self.data_path is None:
            return

        if self.X_train is not None:
            print("------------------------- X train data ------------------------")
            print(self.X_train.head())
            print(self.X_train.info())
            print("\n")

        if self.y_train is not None:
            print("------------------------- Y train data ------------------------")
            print(self.y_train.head())
            print(self.y_train.info())
            print("\n")

        if self.X_test is not None:
            print("------------------------- X test data -------------------------")
            print(self.X_test.head())
            print(self.X_test.info())
            print("\n")

def main():
    if len(sys.argv) < 2:
        print("Please provide path to the directory of raw data.")
        return 0
    path = sys.argv[1]

    step = 0
    csv = ""
    if len(sys.argv) > 2:
        step = int(sys.argv[2])
        if step not in {1,2,3}:
            print("Unable to parse step input: {}".format(step))
            step = 0
        else:
            print("Request performing preprocessing step {}".format(step))
            if step > 1:
                if len(sys.argv) < 4:
                    print("Please provide the file of preprossed data.")
                    return 0
                else:
                    csv = sys.argv[3]

    t_prep = TextPreProcessing(path)
    match step:
        case 0:
            t_prep.preprocess_step1()
            t_prep.preprocess_step2()
            t_prep.preprocess_step3()
            t_prep.save_csv()
        case 1:
            t_prep.preprocess_step1(True)
        case 2:
            t_prep.preprocess_step2(csv, True)
        case 3:
            t_prep.preprocess_step3(csv, True)

if __name__=="__main__":
    main()