import numpy as np
from dataclasses import dataclass

@dataclass
class regression_lineaire:
    coefficients: np.ndarray
    features_names: list


def regression(X, Y, features_names):
    """
    Application de la formule du coefficient de corrélation beta trouvé par la minimisation de la SCR.
    Beta = (X'X)^(-1) X'Y.
    Cette fonction retourne un objet "regression_lineaire" contenant les coefficients beta et les noms des variables explicatives.
    """
    Xt_X = np.dot(X.T, X)
    Xt_Y = np.dot(X.T, Y)
    Beta = np.dot(np.linalg.inv(Xt_X), Xt_Y)
    return regression_lineaire(coefficients=Beta, features_names=features_names)