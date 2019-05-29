import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoLarsIC
from sklearn import datasets

# plot AIC
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

def aic(df_data):
    X_data_pd = df_data.toPandas()  # spark DF to pd_df
    X1 = X_data_pd.values
    y1 = X_data_pd['label'].values * 10000

    rng = np.random.RandomState(42)
    X = np.c_[X, rng.randn(X.shape[0], 14)]  # add some bad features
    X1 = np.c_[X1, rng.randn(X1.shape[0], 14)]
    # normalize data as done by Lars to allow for comparison
    X /= np.sqrt(np.sum(X ** 2, axis=0))
    X1 /= np.sqrt(np.sum(X1 ** 2, axis=0))
    # #############################################################################
    # LassoLarsIC: least angle regression with BIC/AIC criterion
    model_aic = LassoLarsIC(criterion='aic')

    t3 = time.time()
    model_aic.fit(X1, y1)
    t_aic = time.time() - t3
    alpha_aic_ = model_aic.alpha_


    def plot_ic_criterion(plot_model, name, color):
        alpha_ = plot_model.alpha_
        alphas_ = plot_model.alphas_
        criterion_ = plot_model.criterion_
        plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
                 linewidth=3, label='%s criterion' % name)
        plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
                    label='alpha: %s estimate' % name)
        plt.xlabel('-log(alpha)')
        plt.ylabel('criterion')


    plt.figure()
    plot_ic_criterion(model_aic, 'AIC', 'b')
    plt.legend()
    plt.title('Information-criterion for model selection (training time %.3fs)'
              % t_aic)
    plt.show()
