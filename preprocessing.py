from pandas import read_csv
from statsmodels.api import add_constant
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
