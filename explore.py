from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from preprocessing import Preprocessing

class ExploratoryDataAnalysis(Preprocessing):

    def summary(self):
        '''
        describe features
        '''
        pass

    def pca_decomposition(self, n_components=5):
        '''
        dimension reduction : find inputs that affect variance the most
        '''
        scaler = StandardScaler()
        scaler.fit(self.xs)
        _xs = scaler.transform(self.xs)

        pca = PCA(n_components=n_components)
        fit = pca.fit(_xs)

        components = fit.components_
        var = fit.explained_variance_ratio_
        return components, var

    def plot(self):
        '''
        create plot based on important features
        '''
        components, var = self.pca_decomposition()
        n_components = len(components)

        ax = plt.figure().add_subplot(111)
        ax.scatter([_ for _ in range(n_components)], var, 
                   label='total explained variance: %0.2f%%\n \
                          n_components: %d' % (100*sum(var), n_components))

        ax.set_title('PCA Analysis')
        ax.set_xlabel('components')
        ax.set_ylabel('explained variance')
        ax.legend()
        plt.show()
