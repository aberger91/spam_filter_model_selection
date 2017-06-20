from sklearn.model_selection import cross_val_score    
from sklearn.neighbors import KNeighborsClassifier
from spam_feature_labels import features
from prepare_spambase_data import standardized_split

def optimize_k(X_train, Y_train):
    # subsetting just the odd ones
    neighbors = list(filter(lambda x: x % 2 != 0, list(range(50))))

    # empty list that will hold cv scores
    cv_scores = []

    # perform 10-fold cross validation
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

    # changing to misclassification error
    MSE = [1 - x for x in cv_scores]

    # determining best k
    optimal_k = neighbors[MSE.index(min(MSE))]
    print("The optimal number of neighbors is %d" % optimal_k)

if __name__ == '__main__':
    k_nearest_neighbors_spam_filter()
