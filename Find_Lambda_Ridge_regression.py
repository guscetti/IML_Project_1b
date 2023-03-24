import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge


def fit(X, y, lam):
    """
    This function receives training data points, then fits the ridge regression on this data
    with regularization hyperparameter lambda. The weights w of the fitted ridge regression
    are returned.

    Parameters
    ----------
    X: matrix of floats, dim = (135,13), inputs with 13 features
    y: array of floats, dim = (135,), input labels)
    lam: float. lambda parameter, used in regularization term

    Returns
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression
    """
    # w = np.zeros((13,))
    # TODO: Enter your code here
    #print(X)
    X_t = X.transpose()
    y_t = y.transpose()
    XX_t = np.dot(X_t, X)
    id = np.identity(len(XX_t))
    w = np.dot(np.linalg.inv(XX_t + lam * id), np.dot(X_t, y))
    return w


def calculate_RMSE(w, X, y):
    """This function takes test data points (X and y), and computes the empirical RMSE of
    predicting y from X using a linear model with weights w.

    Parameters
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression
    X: matrix of floats, dim = (15,13), inputs with 13 features
    y: array of floats, dim = (15,), input labels

    Returns
    ----------
    RMSE: float: dim = 1, RMSE value
    """
    y_pred = np.dot(X, w)
    RMSE = np.sqrt(sum((y_pred - y) ** 2) / len(y))
    assert np.isscalar(RMSE)
    return RMSE


def average_LR_RMSE(X, y, lambdas, n_folds):
    """
    Main cross-validation loop, implementing 10-fold CV. In every iteration (for every train-test split), the RMSE for every lambda is calculated,
    and then averaged over iterations.

    Parameters
    ----------
    X: matrix of floats, dim = (150, 13), inputs with 13 features
    y: array of floats, dim = (150, ), input labels
    lambdas: list of floats, len = 5, values of lambda for which ridge regression is fitted and RMSE estimated
    n_folds: int, number of folds (pieces in which we split the dataset), parameter K in KFold CV

    Returns
    ----------
    avg_RMSE: array of floats: dim = (5,), average RMSE value for every lambda
    """
    RMSE_mat = np.zeros((n_folds, len(lambdas)))


    # TODO: Enter your code here. Hint: Use functions 'fit' and 'calculate_RMSE' with training and test data
    # and fill all entries in the matrix 'RMSE_mat'
    for j in range(len(lambdas)):
        kf = KFold(n_splits=n_folds, shuffle=True)
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            w = fit(X[train_index], y[train_index], j)
            #print(w)
            RMSE_mat[i, j] = (calculate_RMSE(w, X[test_index], y[test_index]))




    avg_RMSE = np.mean(RMSE_mat, axis=0)

    assert avg_RMSE.shape == (5,)
    return avg_RMSE


def transform_data(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component of X)
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant features: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: array of floats: dim = (700,21), transformed input with 21 features
    """
    X_transformed = np.zeros((700, 21))
    # TODO: Enter your code here



    for i in range(len(X)):
        for j in range(0, 5):
            X_transformed[i, j] = X[i, j]
            X_transformed[i, j + 5] = X[i, j] ** 2
            X_transformed[i, j + 10] = np.exp(X[i, j])
            X_transformed[i, j + 15] = np.cos(X[i, j])
        X_transformed[i, 20] = 1

    assert X_transformed.shape == (700, 21)
    return X_transformed


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    # print(data.head())

    X = data.to_numpy()
    # The function calculating the average RMSE
    lambdas = [0.1, 10, 50, 100, 300]
    n_folds = 10
    avg_RMSE = average_LR_RMSE(transform_data(X), y, lambdas, n_folds)
    # Save results in the required format
    np.savetxt("./results.csv", avg_RMSE, fmt="%.12f")
