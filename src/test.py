
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import joblib


from config.path_config import path_config
from config.models_gridsearch_config import classifiers,gridsearches


class NBAevaluator():
    def __init__(self) -> None:
        self.gridsearches = gridsearches
        self.classifiers = classifiers

    def score_classifier(self, dataset, classifier, classifier_name, labels, gridsearch=None):
        """
        performs 3 random trainings/tests to build a confusion matrix and prints results with precision and recall scores
        :param dataset: the dataset to work on
        :param classifier: the classifier to use
        :param labels: the labels used for training and validation
        :return:
        """
        if gridsearch is not None:
            # Here we optimize for accuracy for a reason : optimizing for recall yields
            # best classifiers of recall 1 which always classify as a 1 (of course)
            # overfitting on the 1 class. So, we optimize for accuracy and choose the
            # best classifier for recall later on
            gs = GridSearchCV(estimator=classifier,
                              param_grid=gridsearch, scoring="accuracy")
        else:
            gs = classifier

        kf = KFold(n_splits=10, random_state=50, shuffle=True)
        confusion_mat = np.zeros((2, 2))
        recall, precision, accuracy = 0, 0, 0
        for training_ids, test_ids in kf.split(dataset):
            training_set = dataset[training_ids]
            training_labels = labels[training_ids]
            val_set = dataset[test_ids]
            val_labels = labels[test_ids]

            gs.fit(training_set, training_labels)

            if gridsearch is not None:
                classifier = gs.best_estimator_
            else:
                classifier = gs

            predicted_labels = classifier.predict(val_set)
            confusion_mat += confusion_matrix(val_labels, predicted_labels)
            recall += recall_score(val_labels, predicted_labels)
            precision += precision_score(val_labels, predicted_labels)
            accuracy += accuracy_score(val_labels, predicted_labels)
        recall /= 10
        precision /= 10
        accuracy /= 10
        return {
            'confusion_matrix': confusion_mat,
            'recall': recall,
            'precision': precision,
            'accuracy': accuracy,
            'model': classifier}

    def score_classifier_on_test_set(self, test_set, test_labels, classifier, classifier_name):
        predicted_labels = classifier.predict(test_set)
        confusion_mat = confusion_matrix(test_labels, predicted_labels)
        recall = recall_score(test_labels, predicted_labels)
        precision = precision_score(test_labels, predicted_labels)
        accuracy = accuracy_score(test_labels, predicted_labels)
        print(classifier_name + ':')
        print('confusion matrix: \n', confusion_mat)
        print(
            f'recall : {recall} - precision : {precision} - accuracy : {accuracy}')
        return {"confusion_matrix": confusion_mat,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy}

    def load(self):
        return pd.read_csv(path_config['data_path'])

    def load_and_clean(self):
        # Load dataset
        df = self.load()

        # extract names, labels, features names and values
        names = df['Name'].values.tolist()  # players names
        y = df['TARGET_5Yrs'].values  # labels
        paramset = df.drop(['TARGET_5Yrs', 'Name'], axis=1).columns.values
        X = df.drop(['TARGET_5Yrs', 'Name'], axis=1).values

        # replacing Nan values (only present when no 3 points attempts have been performed by a player)
        for x in np.argwhere(np.isnan(X)):
            X[x] = 0.0

        return names, X, y

    def split_train_test(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42)
        return X_train, X_test, y_train, y_test


    def fit_classifiers_(self, dataset, labels, gs):
        # note that the parameters for the vanilla classifiers were adapted to 
        # the best found parameters using the grid search. However, we leave the
        # option of redoing the grid search should you want to refit the models 
        # or change the split. (note : takes about 45mins to do the full gridsearch)
        
        if not gs:
            gridsearches = None
        else:
            gridsearches = self.gridsearches

        records = {}
        for classifier in self.classifiers.keys():
            print(f'fitting: {classifier}')
            if gridsearches is not None:
                gridsearch = gridsearches[classifier]
            else:
                gridsearch = None

            records[classifier] = self.score_classifier(
                dataset, self.classifiers[classifier], classifier, labels, gridsearch)
        return records

    def scale_train_test(self, X_train, X_test):
        MMS = MinMaxScaler()
        X_train = MMS.fit_transform(X_train)
        self.scaler = MMS
        X_test = self.scaler.transform(X_test)
        return X_train, X_test

    def fitting_pipeline(self, gs=False):
        print('loading and cleaning dataset')
        names, X, y = self.load_and_clean()
        # normalize dataset
        # NO SCALING ON ALL DATA => INFO LEAKAGE
        # X = MinMaxScaler().fit_transform(df_vals)

        print('splitting into test and train set')
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test(X, y)
        print("scaling train data and applying on test data")
        self.X_train, self.X_test = self.scale_train_test(self.X_train, self.X_test)
        print('fitting classifiers on train set')
        train_records = self.fit_classifiers_(
            self.X_train, self.y_train, gs=gs)
        self.train_records = train_records
        print('scoring best classifiers on test set')
        test_records = {}
        for record in train_records.keys():
            test_records[record] = self.score_classifier_on_test_set(
                self.X_test, self.y_test, train_records[record]['model'], record)
        
        return test_records

    def select_save_best_model(self,model_name = "logreg"):
        print("performance of best selected model on test set: \n \n")
        self.score_classifier_on_test_set(self.X_test,self.y_test, self.train_records[model_name]['model'], model_name)
        pipeline = Pipeline([('scaler', self.scaler),('model',self.train_records[model_name]['model'])])
        joblib.dump(pipeline,f'nba_performance_prediction_back/pipelines/best_model.pkl')


