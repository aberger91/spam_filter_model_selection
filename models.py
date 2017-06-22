from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
    
class Models:
    '''
    static wrapper over scikit-learn objects
    keyward arguments to be tweaked manually or optimized 
    '''
    def get_models(models=['all']):
        _models = [Models.__dict__[m] for m in Models.__dict__.keys() \
                      if "_spam_filter" in m
                      ]
        for m in models:
            if m == 'all':
                return _models

    def random_forest_spam_filter():
        model = RandomForestClassifier(n_estimators=100, 
                                       max_features=57)
        return model

    def k_nearest_neighbors_spam_filter(n_neighbors=1):
        model = KNeighborsClassifier(n_neighbors=n_neighbors, 
                                     weights='distance')
        return model

    def neural_network_spam_filter():
        model = MLPClassifier(hidden_layer_sizes=(57, 57, 57), 
                              max_iter=1000)
        return model

    def bernoulli_bayes_spam_filter():
        model = BernoulliNB()
        return model

    def gaussian_bayes_spam_filter():
        model = GaussianNB()
        return model

    def support_vector_spam_filter():
        model = svm.SVC()
        return model

    def logit_spam_filter():
        model = LogisticRegression()
        return model
