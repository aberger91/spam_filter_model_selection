from statsmodels.api import add_constant
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Preprocessing():
    '''
    handles missing values and stores basic information about features
    '''
    def __init__(self, 
                 data,    # type: pd.DataFrame
                 ):
        self.xs = getattr(data, 'xs')
        self.ys = getattr(data, 'ys')
        self._initial()

    def _initial(self):
        '''
        remove missing values, print dimensions and features
        '''
        self.xs = self.xs.dropna()
        self.ys = self.ys.dropna()
        print('Dimensions of X: [%d, %d]' % (
              len(self.xs), len(self.xs.ix[0]))
              )
        print('Features\n%s' % self.xs.columns)
        print('Target\n%s' % self.ys.name)
        
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
