import numpy as np


class LinearRegression:
    """
    # w: np.ndarray
    # b: float
    """

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the data on the model.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The label data.

        Returns:
            None
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        self.w = self.w[1:]
        self.b = self.w[0]
        print(self.w.shape, self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X.dot(self.w) + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Fit the data on the model.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The label data.
            lr (float): Learning rate.
            epochs(int): No of iterations.

        Returns:
            None
        """
        b = 0
        w = np.zeros(X.shape[1])
        n = X.shape[0]
        for _ in range(epochs):
            b_gradient = -2 * np.sum(y - X.dot(w) + b) / n
            w_gradient = -2 * np.sum(X.T.dot(y - (X.dot(w) + b))) / n
            b = b + (lr * b_gradient)
            w = w - (lr * w_gradient)
        self.w, self.b = w, b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X.dot(self.w) + self.b
