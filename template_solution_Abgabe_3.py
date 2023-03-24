# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd



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

    #print(X_transformed)

    assert X_transformed.shape == (700, 21)
    return X_transformed


def fit(X, y, lam):
    """
    This function receives training data points, transform them, and then fits the linear regression on this 
    transformed data. Finally, it outputs the weights of the fitted linear regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of floats, dim = (700,), input labels)

    Returns
    ----------
    w: array of floats: dim = (21,), optimal parameters of linear regression
    """
    #w = np.zeros((21,))
    #X_transformed = transform_data(X)
    # TODO: Enter your code here
    Phy = transform_data(X)

    X_t = Phy.transpose()
    y_t = y.transpose()
    XX_t = np.dot(X_t, Phy)
    id = np.identity(len(XX_t))
    w = np.dot(np.linalg.inv(XX_t + lam * id), np.dot(X_t, y))
    return w


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")


    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    #print(data.head())

    X = data.to_numpy()
    #print(X)
    # The function retrieving optimal LR parameters
    w = fit(X, y, 300)
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")
    np.savetxt("./x_values.csv", X, fmt="%.12f")
