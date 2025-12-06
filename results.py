import numpy as np

# 3eme partie : résultats
class results:
    def __init__(self, X, Y, df, Beta):
        """
          Regroupement au sein d'une classe des résultats de la régression et les métriques associées.
        """
        self.X = X
        self.Y = Y
        self.df = df
        self.Beta = Beta
        self.valeurs_prédites = None
        self.erreurs = None
        self.R2 = None
        self.RMSE = None
        self.MSE = None
        self.SCR=None



    def predict(self):
        """
        Calcul des valeurs prédites par le modèle à partir de X et beta puis des résidus.
        """
        self.valeurs_prédites = np.dot(self.X, self.Beta)
        self.erreurs = self.Y - self.valeurs_prédites

    def metriques(self):
        """
        Calcul des deux métriques R² tel que R² = 1 - SCR / SCT et  RMSE tel que RMSE = sqrt(SCR)
        """
        ybar = np.mean(self.Y)
        SCT = np.sum((self.Y - ybar) ** 2)
        self.SCR = np.sum(self.erreurs ** 2)
        self.R2 = (SCT - self.SCR) / SCT
        self.MSE = self.SCR/len(self.Y)
        self.RMSE = np.sqrt(self.MSE)



    def extend_df(self):
        """
        Ajout des colonnes "caleurs prédites" et "erreurs" au DataFrame df.
        """
        self.df["valeurs_prédites"] = self.valeurs_prédites
        self.df["erreurs"] = self.erreurs
        return self.df