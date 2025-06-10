#!/usr/bin/env python3
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import joblib

MAX_VECTORIZATION_FEATURE = 3000
SaveDirName="results/simple_textual_model"

class SimpleModel(object):
    dataset = None
    save_path = None
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
        self.tokenize()
        self.encode()

    def tokenize(self):
        # initialize vectorizer
        vectorizer = TfidfVectorizer(max_features=MAX_VECTORIZATION_FEATURE,    # limit featrues to reduce memory usage
                                    stop_words='english',                       # remove common stopwords
                                    min_df=5,                                   # appear at least in 5 docs
        )
        # fit and transform the documents
        tfidf_matrix = vectorizer.fit_transform(self.dataset['text'])
        self.tfidf_matrix = tfidf_matrix

        # get feature names (tokens)
        tokens = vectorizer.get_feature_names_out()

        # convert to DataFrame for readability
        df_tokens = pd.DataFrame(tfidf_matrix.toarray(), columns=tokens)

        # print top 10 most frequent tokens
        token_scores = tfidf_matrix.sum(axis=0).A1  # flatten to 1D array
        token_freq = dict(zip(tokens, token_scores))
        top_10 = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        for token, score in top_10:
            print(f"{token}: {score:.2f}")

        # save to the original dataset
        self.dataset = pd.concat([self.dataset, df_tokens.set_index(self.dataset.index)], axis=1)
        print(self.dataset.head())

    def encode(self):
        # preparation of train/test data:
        self.y = self.dataset['prdtypecode']
        self.X = self.dataset.drop(['raw_text', 'raw_translation', 'text', 'prdtypecode'], axis=1)

        # encode y data, by replacing the list of product type code with range(N)
        target_length = len(self.y.value_counts().index)
        self.index = self.y.value_counts().index.sort_values()
        self.y.replace(to_replace=self.index, value=range(0,target_length), inplace=True)

        # train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=27)

    def evaluate(self, model, X_test, y_test):
        # predict
        y_pred = model.predict(X_test)
        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(cm, index=[f"Actual {label}" for label in self.index],
                         columns=[f"Predicted {label}" for label in self.index])
        print(df_cm)
        # classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.rename(index=dict(zip(df_report.index, self.index)), inplace=True)
        print(df_report.round(2))
        return df_cm, df_report

    def save_results(self, model, df_cm, df_report, modeltype, compressed=False):
        if compressed:
            joblib.dump(model, os.path.join(self.save_path, modeltype+'_model.pkl'), compress=9)
        else:
            joblib.dump(model, os.path.join(self.save_path, modeltype+'_model.pkl'))
        df_cm.to_csv(os.path.join(self.save_path, modeltype+'_cm.csv'))
        df_report.to_csv(os.path.join(self.save_path, modeltype+'_report.csv'))

    def train_svm(self):
        # initialize linear SVM classifier
        clf = svm.LinearSVC(C=0.1, loss='squared_hinge', dual=True, verbose=True)

        # define parameter grid
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'loss': ['squared_hinge', 'hinge'],
        }

        # run grid search
        grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
        grid.fit(self.X_train, self.y_train)

        print("Best parameters:", grid.best_params_)
        print("Best score:", grid.best_score_)

        return grid.best_estimator_

    def train_rf(self):
        rf = RandomForestClassifier(n_estimators=100, random_state=27, n_jobs=1, verbose=1)
        # Define the hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
        }
        grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
        grid.fit(self.X_train, self.y_train)

        print("Best parameters:", grid.best_params_)
        print("Best score:", grid.best_score_)

        return grid.best_estimator_

    def train_xgb(self):
        model = xgb.XGBClassifier(n_estimators=200,
                                  max_depth=None,
                                  random_state=27,
                                  verbosity=2
        )
        model.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=True)
        return model

    def train(self, modeltype: str = "SVM"):
        if self.dataset is None or self.X_train is None or self.y_train is None:
            print("Please load and encode the data before training a model")
            return
        print(f"Training a simple model with type {modeltype}")
        if modeltype=="SVM":
            model = self.train_svm()
            df_cm, df_report = self.evaluate(model, self.X_test, self.y_test)
        elif modeltype=="RF":
            model = self.train_rf()
            df_cm, df_report = self.evaluate(model, self.X_test, self.y_test)
        elif modeltype=="XGB":
            model = self.train_xgb()
            df_cm, df_report = self.evaluate(model, self.X_test, self.y_test)
        else:
            print(f'{modeltype} as model is not supported')
        
        if modeltype=="RF":
            self.save_results(model, df_cm, df_report, "RF", compressed=True)
        else:
            self.save_results(model, df_cm, df_report, modeltype)

    def visualize(self):
        # reduce the dimension of TF-IDF matrix for visualization
        svd = TruncatedSVD(n_components=2, random_state=27)
        reduced_matrix = svd.fit_transform(self.tfidf_matrix)

        # plot the reduced TF-IDF matrix
        df_plot = pd.DataFrame(reduced_matrix, columns=['Component 1', 'Component 2'])
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_plot, x='Component 1', y='Component 2')
        plt.title("2D Projection of TF-IDF Matrix via Truncated SVD")
        plt.show()

def main():
    if len(sys.argv) < 2:
        print("Please provide path to the preprocessed data.")
        return 0
    file = sys.argv[1]

    modeltype = "SVM"
    if len(sys.argv) > 2:
        modeltype = sys.argv[2].upper()
        if not modeltype in {"SVM", "RF", "XGB"}:
            print(f"Model type {modeltype} is not supported, please select among 'SVM', 'RF' and 'XGB'.")
            return 0

    model = SimpleModel(file)
    try:
        model.train(modeltype)
        model.visualize()
    except KeyboardInterrupt:
        print("Interrupt by keyboard")

if __name__=="__main__":
    main()