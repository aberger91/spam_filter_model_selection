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
from sklearn.metrics import (
                            confusion_matrix, 
                            classification_report, 
                            accuracy_score,
                            mean_squared_error
                            ) 
from sklearn.model_selection import cross_val_score
from explore import ExploratoryDataAnalysis
from preprocessing import Preprocessing
from models import Models
import numpy as np


class ModelComparison(Preprocessing):
    '''
    '''
    metrics = (
               confusion_matrix, 
               classification_report, 
               accuracy_score, 
               mean_squared_error
               )

    def __init__(self, 
                 path_to_data,  # type: str
                 features,      # type: List[str]
                 target,        # type: str
                 sep=','
                 ):
        '''
        cross_validation_scores -> Dict[str]: List[float]
        '''
        super().__init__(path_to_data, features, target, sep=sep)
        self.cross_validation_scores = {}

    def get_predictions(self, 
                        model, 
                        output_as_probability=False, 
                        threshold=0.5):
        '''
        train a model and gather prediction results
        model -> sklearn model object
        output_as_probability -> boolean
        threshold -> float
        '''
        X_train, X_test, Y_train, Y_test = self.get_standardized_training_test_split()
        fit = model.fit(X_train, Y_train)
        if output_as_probability:
            predictions = [1 if x > threshold else 0 for x in fit.predict(X_test)]
        else:
            predictions = model.predict(X_test)
        return (X_train, Y_train, fit, Y_test, predictions)

    def execute(self):
        '''
        populate self.cross_validation_scores and display model results
        '''
        for model in Models.get_models():
            print('=' * 56)
            name = model.__name__
            print('%s model summary' % name.upper())
            print('=' * 56)

            x_train, y_train, fitted_model, y_true, y_hats = self.get_predictions(model())

            for metric in self.metrics:
                result = metric(y_true, y_hats)

                if isinstance(result, float):
                    print(' %s %s: %0.4f ' % (name, metric.__name__.upper(), result))
                else:
                    print(' %s %s ' % (name, metric.__name__.upper()))
                    print(result)

            cv_scores = cross_val_score(model(), x_train, y_train, cv=10, scoring='accuracy')
            print("CROSS VALIDATION RESULTS\n%s" % np.mean(cv_scores))
            self.cross_validation_scores[name] = cv_scores

            print('*' * 56)


if __name__ == '__main__':
    from feature_labels import features
    args = ['data/spambase.data', features, 'is_spam']

    #explore = ExploratoryDataAnalysis(*args)
    #explore.pca_decomposition()

    model_comparison = ModelComparison(*args)
    model_comparison.execute()
