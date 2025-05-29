#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

SaveDirName="results_bert"
MODEL_NAME = "distilbert-base-uncased"
NUM_SAMPLES= 500
DOWNSAMPLE = False

class BertModel(object):
    dataset = None
    save_path = None
    trainer = None
    def __init__(self, file):
        # read data from preprocessed file
        try:
            self.dataset = pd.read_csv(file, index_col=0, delimiter=';')
        except:
            print("Please check the file provided")
            return
        path = os.path.dirname(os.path.abspath(__file__))
        self.save_path = os.path.join(path, SaveDirName)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.encode()
        num_labels = self.y.value_counts().shape[0]
        print(f"Number of labels to be classified: {num_labels}")
        print(f"Training model: {MODEL_NAME}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
        except Exception as e:
            print(f"Failed loading pretrained Bert model, error message: {e}")

    def encode(self):
        if DOWNSAMPLE:
            # down-sample the dataset to be able to train the bert model within acceptable time
            target_index = self.dataset['prdtypecode'].value_counts().index
            downsampled_dataset = pd.DataFrame()
            for idx in target_index:
                samples = self.dataset[self.dataset['prdtypecode']==idx].sample(n=NUM_SAMPLES, random_state=27)
                downsampled_dataset = pd.concat([downsampled_dataset, samples], axis=0)
            downsampled_dataset.sort_index(inplace=True)
        else:
            downsampled_dataset = self.dataset
        print(downsampled_dataset['prdtypecode'].value_counts())

        # preparation of train/test data:
        self.y = downsampled_dataset['prdtypecode']
        self.X = downsampled_dataset['text']

        # encode y data, by replacing the list of product type code with range(N)
        target_length = len(self.y.value_counts().index)
        self.index = self.y.value_counts().index.sort_values()
        self.y.replace(to_replace=self.index, value=range(0,target_length), inplace=True)

        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=27)

        # prepare datasets for training/evaluating with pretrained bert model
        self.train_dataset = Dataset.from_dict({"text": X_train, "label": y_train})
        self.eval_dataset = Dataset.from_dict({"text": X_test, "label": y_test})

    def tokenize_function(self, batch):
        return self.tokenizer(batch['text'], padding="max_length", truncation=True, max_length=128)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        return {"accuracy": acc, "f1": f1}

    def train(self):
        # tokenize
        self.train_dataset = self.train_dataset.map(self.tokenize_function, batched=True)
        self.eval_dataset = self.eval_dataset.map(self.tokenize_function, batched=True)

        # set pytorch format
        self.train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        self.eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        # training arguments
        training_args = TrainingArguments(
            output_dir=self.save_path,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            num_train_epochs=3,
            weight_decay=0.01,
            save_steps=500,
            logging_dir=self.save_path,
            logging_steps=10,
            load_best_model_at_end=True,
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=data_collator,
        )

        # train the model
        self.trainer.train()

    def evaluate(self):
        # evaluate
        eval_result = self.trainer.evaluate()
        print(eval_result)

        # make predictions
        predictions = self.trainer.predict(self.eval_dataset)
        y_true = predictions.label_ids
        y_pred = np.argmax(predictions.predictions, axis=1)

        # confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cm, index=[f"Actual {label}" for label in self.index],
                         columns=[f"Predicted {label}" for label in self.index])
        print(df_cm)
        # classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.rename(index=dict(zip(df_report.index, self.index)), inplace=True)
        print(df_report.round(2))
        return df_cm, df_report

    def save_results(self, df_cm, df_report):
        path = os.path.join(self.save_path, "saved_model")
        if not os.path.exists(path):
            os.makedirs(path)
        df_cm.to_csv(os.path.join(self.save_path, 'bert_cm.csv'))
        df_report.to_csv(os.path.join(self.save_path, 'bert_report.csv'))
        self.trainer.save_model(path)


def main():
    if len(sys.argv) < 2:
        print("Please provide path to the preprocessed data.")
        return 0
    file = sys.argv[1]

    bert = BertModel(file)
    try:
        bert.train()
        df_cm, df_report = bert.evaluate()
        bert.save_results(df_cm, df_report)
    except Exception as e:
        print(f"Failed training Bert model, error message: {e}")


if __name__=='__main__':
    main()