from sklearn.metrics import (
                            confusion_matrix, 
                            classification_report, 
                            accuracy_score,
                            mean_squared_error
                            ) 
from sklearn.model_selection import cross_val_score
from preprocessing import Preprocessing
from models import Models
import numpy as np

class MLModelComparison(Preprocessing):
    '''
    '''
    metrics = (
               # confusion_matrix, 
               # classification_report, 
               accuracy_score, 
               mean_squared_error
               )

    def __init__(self, 
                 data,  # type: pd.DataFrame
                 models=['all']
                 ):
        '''
        cross_validation_scores -> Dict[str]: List[float]
        '''
        super().__init__(data)
        self._models = Models.get_models(models=models)
        self.scores = {}

    @property
    def models(self):
        return self._models

    def get_predictions(self, 
                        model,                       #  type: sklearn
                        output_as_probability=False, #  type: bool
                        threshold=0.5                #  type: float
                        ):
        '''
        train a model and gather prediction results
        return -> Tuple[numpy.array[float]] # (x_train, y_train, fitted_model, y_true, y_hats)
        '''
        X_train, X_test, Y_train, Y_test = self.get_standardized_training_test_split()
        fit = model.fit(X_train, Y_train)

        if output_as_probability:
            predictions = [1 if x > threshold else 0 for x in fit.predict(X_test)]
        else:
            predictions = fit.predict(X_test)
        return (X_train, Y_train, fit, Y_test, predictions)
    
    def execute(self, cv=3):
        '''
        populate self.scores
        '''
        for model in self.models:
            name = model.__name__
            _results = self.get_predictions(model())
            x_train, y_train, fitted_model, y_true, y_hats = _results

            self.scores[name] = {}
            for metric in self.metrics:
                metric_name = metric.__name__.upper()
                self.scores[name][metric_name] = metric(y_true, y_hats)

            cv_scores = cross_val_score(model(), 
                                        x_train, 
                                        y_train, 
                                        cv=cv, 
                                        scoring='accuracy')
            self.scores[name]['CROSS_VALIDATION'] = np.mean(cv_scores)
        return self.scores
