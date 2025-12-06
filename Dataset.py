import numpy as np
import pandas as pd

# Partie 1

class Dataset:
    """
    Initialisation d'un objet "Dataset" contenant des matrices X (variables explicatives n*p)
    et Y (variable expliquée n*1 ) et d'une liste "features_names" (noms des variables explicatives).

    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, features_names: list):
        """
        Initialisation d'un objet "Dataset" contenant des matrices X (variables explicatives n*p)
        et Y (variable expliquée n*1 ) et d'une liste "features_names" (noms des variables explicatives).

        """
        self.X = X
        self.Y = Y
        self.features_names = features_names
        self.n = X.shape[0]
        self.p = X.shape[1]

    def add_intercept(self):
        """
        Concaténation à gauche d'une colonne de 1 dans la matrice X pour ajouter un intercept dans le modèle.
        Et ajout de "constate" en début de liste.
        """
        self.X = np.column_stack([np.ones((self.n, 1)), self.X])
        self.features_names.insert(0, "constante")

    def transform_to_dataframe(self):
        """
        Transformation des données de "Dataset" en DataFrame (pandas) pour son utilisation dans la partie 2.
        """
        df = pd.DataFrame(self.X, columns=self.features_names)
        df["variable_expliquée"] = self.Y
        return df