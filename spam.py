'''
#Spam Filtering with Classifiers
  - ordinary least squares regression
  - logistic regression
  - naive bayesian classifiers
  - multilayer perceptron neural network

*References*:
  - https://archive.ics.uci.edu/ml/datasets/Spambase
  - Mark Hopkins, Erik Reeber, George Forman, Jaap Suermondt
    Hewlett-Packard Labs, 1501 Page Mill Rd., Palo Alto, CA 94304
'''
from pandas import read_csv
from statsmodels.api import add_constant
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from statsmodels.api import Logit
from sklearn.metrics import (
                            confusion_matrix, 
                            classification_report, 
                            accuracy_score,
                            mean_squared_error
                            ) 


class Preprocessing():
    def __init__(self, path_to_data, features, target, sep=','):
        '''
        path_to_data -> str
        features -> list (input labels)
        target -> str (response label)
        sep -> char (delimiter)
        '''
        self.xs = read_csv(path_to_data, names=features, sep=sep)
        #self.xs = self.xs.dropna()
        self.ys = self.xs.pop(target)

    def get_standardized_training_test_split(self, test_size=0.4):
        '''
        split samples into training / testing sets on a unit scale
        '''
        X_train, X_test, Y_train, Y_test = train_test_split(self.xs, 
                                                            self.ys, 
                                                            test_size=test_size)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
        return X_train, X_test, Y_train, Y_test

    def get_training_test_split(self, test_size=0.4):
        '''
        split samples into training / testing sets
        '''
        self.xs = add_constant(self.xs)
        X_train, X_test, Y_train, Y_test = train_test_split(self.xs, 
                                                            self.ys, 
                                                            test_size=test_size)
        return X_train, X_test, Y_train, Y_test


class ExploratoryDataAnalysis(Preprocessing):
    def summary(self):
        '''
        describe features
        '''
        pass

    def pca_decomposition(self):
        '''
        dimension reduction : find inputs that affect variance the most
        '''
        pass

    def plot(self):
        '''
        create plot based on important features
        '''
        pass

    
class Models:
    def get_models():
        models = [Models.__dict__[m] for m in Models.__dict__.keys() if "_spam_filter" in m]
        return models

    def random_forest_spam_filter():
        model = RandomForestClassifier(n_estimators=100, max_features=57)
        return model

    def k_nearest_neighbors_spam_filter(n_neighbors=1):
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
        return model

    def neural_network_spam_filter():
        model = MLPClassifier(hidden_layer_sizes=(57, 57, 57), max_iter=1000)
        return model

    def bernoulli_bayes_spam_filter():
        model = BernoulliNB()
        return model

    def gaussian_bayes_spam_filter():
        model = GaussianNB()
        return model

    def logit_spam_filter():
        model = LogisticRegression()
        return model

    def support_vector_spam_filter(threshold=0.5):
        model = svm.SVC()
        return model


class ModelComparison(Preprocessing):
    metrics = (
               confusion_matrix, 
               classification_report, 
               accuracy_score, 
               mean_squared_error
               )

    def get_predictions(self, model, output_as_probability=False, threshold=0.5):
        X_train, X_test, Y_train, Y_test = self.get_standardized_training_test_split()
        fit = model.fit(X_train, Y_train)
        if output_as_probability:
            predictions = [1 if x > threshold else 0 for x in fit.predict(X_test)]
        else:
            predictions = model.predict(X_test)
        return (fit, Y_test, predictions)

    def execute(self):
        for model in Models.get_models():
            print('=' * 56)
            name = model.__name__
            print('%s model summary' % name.upper())
            print('=' * 56)

            fitted_model, y_true, y_hats = self.get_predictions(model())
            for metric in self.metrics:
                print(' %s %s ' % (name, metric.__name__.upper()))
                print(metric(y_true, y_hats))
            print('*' * 56)


if __name__ == '__main__':
    from spam_feature_labels import features

    models = ModelComparison('../data/spambase.data', features, 'is_spam')
    models.execute()
