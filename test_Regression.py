import numpy as np
import pandas as pd
from sklearn import datasets
import statsmodels.api as sm
from Dataset import Dataset
from Regression import regression
from results import results



def dataset_synthetique():
    """
    Génère un jeu de données artificiel avec make_regression
    Création d'un objet "Dataset" avec p explicatives, qui sera utilisée pour executer les fonctions de test.
    """
    T, V = datasets.make_regression(n_samples=1000, n_features=6, noise=10)
    feature_list = [f"X{i}" for i in range(6)]
    data_1 = Dataset(X=T, Y=V, features_names=feature_list)
    return data_1


def petit_dataset():
    """Crée un petit dataset pour les tests de régression et métriques."""
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    Y = np.array([1, 2, 2, 3])
    feature_names = ["constante", "X1"]
    return X, Y, feature_names


def test_add_intercept():
    """
    Test de la fonction add_intercept de la classe "Dataset".
    Verification que le nombre de lignes ne change pas, que le nombre de colonnes augmente bien de 1,
    que la première colonne est constante et que la liste des noms de colonne est cohérente.
    """
    data_1 = dataset_synthetique()
    data_1.add_intercept()

    assert data_1.X.shape[0] == data_1.n ,f"The number of samples should remain {data_1.n} after adding intercept."
    assert data_1.X.shape[1] == data_1.p + 1 ,f"The number of features should be {data_1.p + 1}. after adding intercept."
    assert data_1.features_names[0] == "constante","The first column name should be 'constante'."
    assert len(data_1.features_names) == data_1.p + 1, "The Feature names list length is incorrect."
    print("test_add_intercept OK")


def test_to_dataframe():
    """
    Test de la fonction to_dataframe de la classe "Dataset" qui permet de transformer en dataframe.
    Vérification que le type de la sortie est bien un dataframe (pandas),
    que le nombre de lignes, le nombre de colonnes (features + intercept + Y)
    et que le noms des colonnes correspondent bien.
    """
    data_1 = dataset_synthetique()
    data_1.add_intercept()

    df = data_1.transform_to_dataframe()

    assert type(df) == pd.DataFrame, "The output should be a pandas DataFrame."
    assert df.shape[0] == data_1.n, f"The number of rows in DataFrame should be {data_1.n}."
    assert df.shape[1] == data_1.p + 2, f"The number of columns in DataFrame should be {data_1.p + 2}."
    assert df.columns[0] == "constante", "The first column name should be 'constante'."
    assert df.columns[-1] == "variable_expliquée", "The last column name should be 'variable_expliquée'."
    print("test_to_dataframe OK")


def test_regression():
    """
    Teste la fonction regression() en comparant les coefficients estimés avec ceux du modèle OLS de statsmodels.
    """
    X, Y, feature_names = petit_dataset()
    reg_model = regression(X, Y, feature_names)
    sm_model = sm.OLS(Y, X).fit()

    assert reg_model.coefficients.shape[0] == X.shape[1], "The number of coefficients should match the number of features"
    assert np.allclose(reg_model.coefficients, sm_model.params), "The coefficients are not computed correctly."
    print("test_regression OK")



def test_predict():
    """
    Test de la cohérence des valeurs prédites.
    Les prédictions calculées avec Beta doivent être identiques à celles du modèle statsmodels.
    """
    # Calculer les valeurs prédites avec nos fonctions
    X, Y, feature_names = petit_dataset()
    reg_model = regression(X, Y, feature_names)
    Beta = reg_model.coefficients
    res = results(X, Y, None, Beta)
    res.predict()
    Y_pred = res.valeurs_prédites
    # Faire de la regression sur le mème échantillon avec statsmodels
    sm_model = sm.OLS(Y, X).fit()
    sm_Y_pred = sm_model.predict(X)

    assert np.allclose(Y_pred, sm_Y_pred), "The predicted values are not computed correctly by the predict() method."
    print("test_predict OK")




def test_metriques():
    """
    Test des métriques R^2 et RMSE calculées dans la classe results.
    Les valeurs obtenues doivent être les mêmes que celles obtenues via statsmodels.
    """
    X, Y, feature_names = petit_dataset()
    reg_model = regression(X, Y, feature_names)
    Beta = reg_model.coefficients

    resultats = results(X, Y, None, Beta)
    resultats.predict()
    resultats.metriques()

    local_R2 = resultats.R2
    local_RMSE = resultats.RMSE

    sm_model = sm.OLS(Y, X).fit()
    R2 = sm_model.rsquared
    RMSE = np.sqrt(np.mean((Y - sm_model.predict(X))**2))

    assert np.allclose(R2, local_R2), "Both R² measures are expected to be equal."
    assert np.allclose(RMSE, local_RMSE), "The two RMSE values must be equal."
    print("test_metriques OK")


def test_extend_df():
    """
    Test de l'extension du dataframe en vérifiant que l'ajout des colonnes "valeurs_prédites" et "erreurs" a bien été effectué.
    Et vérification que le nombre d'observations reste identique.
    """
    X, Y, feature_names = petit_dataset()
    reg_model = regression(X, Y, feature_names)

    df = pd.DataFrame(X, columns=feature_names)
    df["variable_expliquée"] = Y

    res = results(X, Y, df, reg_model.coefficients)
    res.predict()
    extended_df = res.extend_df()

    assert "valeurs_prédites" in extended_df.columns, "The new DataFrame should contain the column of the predicted values"
    assert "erreurs" in extended_df.columns, "The DataFrame should contain errors column."
    assert extended_df.shape[0] == X.shape[0], "The number of rows in the extended DataFrame should match the number of initial samples."
    print("test_extend_df OK")


if __name__ == "__main__":
    test_add_intercept()
    test_to_dataframe()
    test_regression()
    test_predict()
    test_metriques()
    test_extend_df()

    print("\nTous les tests ont réussi.")