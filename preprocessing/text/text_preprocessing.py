#!/usr/bin/env python3
import os
import sys
import re
import pandas as pd
import numpy as np
from langdetect import detect
import html
import tiktoken
from difflib import SequenceMatcher

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
            candidate = X[(X['prdtypecode'].astype(str) == c) & (X['language'].str.upper() != "EN")]
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

    def parse_translation(self, X: pd.DataFrame):
        X['title'] = ""
        X['translation'] = ""
        rows = X.shape[0]
        i = 1
        for idx, text in zip(X.index, X['translated_text']):
            print("Processing row {}/{}".format(i, rows))
            print(f'Original text: {text}')
            i += 1
            text = re.sub(r'\*+|\-', '', text)
            X.loc[idx, 'translated_text'] = text
            match_title = re.search(r'Title:\s*(.*)', text)
            if match_title:
                title = match_title.group(1)
                X.loc[idx, 'title'] = title
                print("Extracted Title:", title)
            else:
                title = text.splitlines()[0]
                X.loc[idx, 'title'] = title
                print("Put default text as title:", title)

            match_trans = re.search(r'Translation:\s*(.*)', text)
            if match_trans:
                translation = match_trans.group(1)
                X.loc[idx, 'translation'] = translation
                print("Extracted translation:", translation)

            if match_trans and match_title:
                if abs(len(re.findall(r'\b\w+\b', title))-len(re.findall(r'\b\w+\b', translation)))<3:
                    similarity = SequenceMatcher(None, title, translation).ratio()
                    if similarity > 0.15 and similarity < 0.85:
                        X.loc[idx, 'title'] = translation
                        print("Use translation {} for text {}".format(translation, title))
        return X

    def clean_text_final(self, X: pd.Series):
        texts = []
        rows = X.shape[0]
        for i, text in enumerate(X):
            # remove unwanted special characters (keeping symbolic letters and whitespace)
            text = re.sub(r'[^a-zA-Z\s\'À-ÿäöüÄÖÜß]', ' ', text, flags=re.UNICODE)
            # remove multiple whitespace characters and the leading/trailing whitespace
            text = re.sub(r'\s+', ' ', text).lstrip().rstrip()
            print("Cleaning row {}/{}".format(i+1, rows))
            texts.append(text)
        return pd.Series(data=texts, index=X.index)

    def preprocess_step4(self, csv=None, save_csv=False):
        if csv is None and self.text_data is None:
            print("Performing the preprocessing steps first or load from a file.")
            return

        if csv is not None:
            try:
                self.text_data = pd.read_csv(os.path.join(self.save_path, csv), index_col=0)
            except FileNotFoundError as e:
                print("Please check the file provided, error message:{}".format(e))
                return

        # check if the loaded data contains preprocessed and translated description
        if not 'translated_text' in self.text_data.columns:
            print("Please provide the data that has been preprocessed and translated.")
            return

        self.text_data = self.parse_translation(self.text_data)

        self.text_data.rename({'description':'raw_text',
                               'title': 'text',
                               'translated_text':'raw_translation'},
                               axis=1, inplace=True)

        self.text_data['text'] = self.clean_text_final(self.text_data['text'])

        self.text_data = self.text_data[['raw_text', 'raw_translation', 'text', 'prdtypecode']]

        if save_csv:
            self.text_data.to_csv(os.path.join(self.save_path, SaveDataFilename+'_final.csv'), sep=';')

    def preprocess_step5(self, csv=None):
        if csv is None:
            print("Please provide the data file.")
            return
        csv_aug = os.path.splitext(csv)[0]+'_augmented.csv'
        try:
            text_data = pd.read_csv(os.path.join(self.save_path, csv), delimiter=';', index_col=0)
            augmented = pd.read_csv(os.path.join(self.save_path, csv_aug), delimiter=';', index_col=0)
        except FileNotFoundError as e:
            print("Please check the file provided, error message:{}".format(e))
            return

        # filter augmented data that has inconsistent textual description
        augmented = augmented.dropna()
        augmented = augmented[augmented['text'].apply(lambda x: len(x.split()))>Filtering_Params['min_words_length']]

        # find the minority class
        target_size = Filtering_Params['target_sample_size']
        minority_class_size = text_data['prdtypecode'].value_counts()[text_data['prdtypecode'].value_counts()<target_size] + \
            augmented['prdtypecode'].value_counts()
        print("Minority class size in total:\n", minority_class_size)
        new_samples = pd.DataFrame()
        dropped_new_samples = pd.DataFrame()
        # drop augmented data for which the total size (including existing samples) is greater than target size
        for idx, val in zip(minority_class_size.index, minority_class_size.values):
            new_data = augmented[augmented['prdtypecode']==idx]
            if val == target_size:
                new_samples = pd.concat([new_samples, new_data], axis=0)
            elif val < target_size:
                print(f"Warning: augmented data is not enough for index {idx} (size: {val}, target size: {target_size})")
                print("Preprocessing aborted, please generate the augmented data again!")
                return
            else:
                diff = val - target_size
                dropped_data = new_data.sample(n=diff, random_state=27)
                new_data = new_data.drop(dropped_data.index)
                new_samples = pd.concat([new_samples, new_data], axis=0)
                dropped_new_samples = pd.concat([dropped_new_samples, dropped_data], axis=0)
        # reset index for augmented data
        last_index = text_data.index[-1]
        new_samples.index = range(last_index+1, last_index+1+new_samples.shape[0])

        # save data
        save_data = pd.concat([text_data, new_samples], axis=0)
        save_data.to_csv(os.path.join(self.data_path, SaveDataFilename+'.csv'), sep=';')

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
        if step not in {1,2,3,4,5}:
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
        case 1:
            t_prep.preprocess_step1(True)
        case 2:
            t_prep.preprocess_step2(csv, True)
        case 3:
            t_prep.preprocess_step3(csv, True)
        case 4:
            t_prep.preprocess_step4(csv, True)
        case 5:
            t_prep.preprocess_step5(csv)

if __name__=="__main__":
    main()