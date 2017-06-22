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
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
from cross_validation import MLModelComparison
from feature_labels import features
import seaborn

class SpamBaseData():
    def __init__(self, 
                 path_to_data,  # type: str
                 features,      # type: List[str]
                 target,        # type: str
                 sep=','
                 ):
        self.xs = read_csv(path_to_data, 
                           names=features, 
                           sep=sep)
        self.ys = self.xs.pop(target)


def main():
    data = SpamBaseData('data/spambase.data', 
                         features, 
                         'is_spam')
    cross_val = MLModelComparison(data, models=['all'])
    results = cross_val.execute(cv=3)  # type : Dict[str] -> Dict[List[float]]]

    results = DataFrame(results).T
    print(results)

    def print_cross_validation_results(results):
        for model in results.keys():
            print(' \t\t %s ' % model.upper())
            for metric, score in results[model].items():
                if isinstance(score, float):
                    print(' %s: %0.4f ' % (
                          metric, score)
                         )
                else:
                    print(metric)
                    print(' %s ' % score)
            print(56 * '*')
            print()
    #print_cross_validation_results(results)

    ax = plt.figure().add_subplot(111)
    results.plot(ax=ax, kind='bar')
    plt.xticks(rotation=65)
    plt.show()
    
    results.plot(subplots=True, kind='bar')
    plt.xticks(rotation=65)
    plt.show()

if __name__ == '__main__':
    main()
