import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch

def make_meshgrid(X_train, X_test, y_train, y_test, h=.02):
    x_min = min(X_train[:, 0].min(), X_test[:, 0].min()) - .5
    x_max = max(X_train[:, 0].max(), X_test[:, 0].max()) + .5
    y_min = min(X_train[:, 1].min(), X_test[:, 1].min()) - .5
    y_max = max(X_train[:, 1].max(), X_test[:, 1].max()) + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    return xx, yy


def predict_proba_on_mesh(clf, xx, yy):
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    return Z

def predict_proba_on_mesh_tensor(clf, xx, yy):
    data_tensor = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    probs = clf.predict_proba_tensor(data_tensor)[:, 1]
    probs = probs.reshape(xx.shape)
    return probs


def plot_predictions(xx, yy, Z, X_train=None, X_test=None, y_train=None, y_test=None,
                     figsize=(10, 10),
                     title="predictions",
                     cm=plt.cm.RdBu,
                     cm_bright=ListedColormap(['#FF0000', '#0000FF'])):
    plt.figure(figsize=figsize)
    plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    
    if X_train is not None:
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                    edgecolors='k')

    if X_test is not None:
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                    edgecolors='k', alpha=0.6)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(title)
    plt.show()
    


