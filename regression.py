import matplotlib.pyplot as plt
from statsmodels.api import Logit
import seaborn as sns
import pandas as pd
from spam_feature_labels import features
from prepare_spambase_data import split

def logit_decision_boundary():
    dat = pd.read_csv('../data/spambase.data', names=features)

    xs = np.array(dat[features])
    ys = dat.pop('is_spam')

    _features = ['word_freq_000', 'word_freq_george']

    X_train, X_test, Y_train, Y_test = train_test_split(xs, ys, test_size=0.4)

    logit_model = LogisticRegression()
    logit_fit = logit_model.fit(X_train, Y_train)

    x_min, x_max = xs[:, 0].min() - .5, xs[:, 0].max() + .5
    y_min, y_max = xs[:, 1].min() - .5, xs[:, 1].max() + .5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))

    f, ax = plt.subplots(figsize=(12, 10))

    Z = logit_fit.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    cs = ax.contourf(xx, yy, Z, cmap='RdBu', alpha=.5)
    cs2 = ax.contour(xx, yy, Z, cmap='RdBu', alpha=.5)

    ax_c = f.colorbar(cs)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize=14)

    ax.plot(X_train[Y_train == 0, 0], X_train[Y_train == 0, 1], 'ro', label=_features[0])
    ax.plot(X_train[Y_train == 1, 0], X_train[Y_train == 1, 1], 'bo', label=_features[1])

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.legend(loc='upper left', scatterpoints=1, numpoints=1)
    plt.show()


#print(logit_fit.summary())
#print('Logit Confidence Intervals\n%s\n' % logit_fit.conf_int())
#
#b0 = logit_fit.params[0] # intercept coeficent
#
#significant_values = logit_fit.pvalues.loc[logit_fit.pvalues < 0.0005]
#coefficients = logit_fit.params.loc[significant_values.index]
#odds_ratios = coefficients.apply(lambda x: 100 * math.exp(x) / (1 + math.exp(x)))
#
#print("Significant Values\n%s\n" % significant_values)
#print("Coefficients\n%s\n" % coefficients)
#print("Odds Ratios\n%s\n" % odds_ratios)

#f = plt.figure()
#ax = f.add_subplot(111)
#ax.scatter(logit_fit.predict(), logit_fit.resid_response)
#ax.set_xlabel('predictions')
#ax.set_ylabel('residuals')
#plt.show()

if __name__ == '__main__':
    logit_decision_boundary()
