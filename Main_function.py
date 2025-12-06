import numpy as np
from sklearn import datasets
from Dataset import Dataset
from Regression2 import regression
from results import results

'''Après avoir défini toute la structure et les classes d'objets, 
nous allons écrire une fonction qui exécute l’ensemble du processus 
et retourne tous les résultats de la régression.'''

'''On va procéder étape par étape.'''
'''Dans un premier temps, nous définissons quelques fonctions 
utilitaires qui vont nous permettre de construire la fonction principale.'''


def generer_donnees():
    T, V = datasets.make_regression(n_samples=1000, n_features=6, noise=100)
    feature_names = [f"X{i}" for i in range(T.shape[1])]
    return T, V, feature_names


def preparer_dataset(T, V, feature_names):
    data_1 = Dataset(X=T, Y=V, features_names=feature_names)
    data_1.add_intercept()
    df = data_1.transform_to_dataframe()
    return data_1, df


def faire_regression(data_1):
    res_regression = regression(data_1.X, data_1.Y, data_1.features_names)
    Beta = res_regression.coefficients
    return res_regression, Beta


def coeff_en_dict(data_1, Beta):
    dict_coeff = {data_1.features_names[i]: Beta[i] for i in range(len(data_1.features_names))}
    return dict_coeff


def analyser_resultats(data_1, df, Beta):
    res = results(data_1.X, data_1.Y, df, Beta)
    res.predict()
    res.metriques()
    df_extended = res.extend_df()
    return res, df_extended

'''
 Définir la fonction principale qui exécute toutes les étapes, et retourne tous les résultats sous
  forme bien tabulée.
'''
def Fonction_regression():

    # Partie données
    T, V, feature_names = generer_donnees()

    # Dataset
    data_1, df = preparer_dataset(T, V, feature_names)
    print(df)

    # Régression
    res_regression, Beta = faire_regression(data_1)
    print("dataclass contenant les coefficients :", res_regression)

    # Dictionnaire
    dict_coeff = coeff_en_dict(data_1, Beta)
    print("Dictionnaire des coefficients : ", dict_coeff)

    # Résultats et métriques
    resultats, df_extended = analyser_resultats(data_1, df, Beta)

    print("Extended DataFrame :\n", df_extended.head())

    # Faire un résumé statistique
    X = data_1.X
    Y = data_1.Y
    n, p = X.shape


    SCR = np.sum(resultats.erreurs ** 2)
    sigma2 = SCR / (n - p)

    # Variance-covariance de Beta
    XtX = np.dot(X.T, X)
    XtX_inv = np.linalg.inv(XtX)
    var_beta = sigma2 * XtX_inv
    std_beta = np.sqrt(np.diag(var_beta))

    # stat de student et p-values
    from scipy.stats import norm
    t = Beta.flatten() / std_beta
    p_values = 2 * (1 - norm.cdf(np.abs(t)))

    # Afficher le summary
    print("\n================ Résumé du modèle ================")
    print(f"Nombre d'observations : {n}")
    print(f"Nombre de variables   : {p}")
    print(f"R²                    : {resultats.R2:.4f}")
    print(f"RMSE                  : {resultats.RMSE:.4f}")
    print("==================================================")

    print(f"{'Variable':<15} {'Coef':>12} {'Std Err':>12} {'t_stat':>10} {'P>|t|':>10}")
    print("-" * 70)

    for i, nom in enumerate(data_1.features_names):
        row = f"{nom:<15} {Beta[i]:12.4f} {std_beta[i]:12.4f} {t[i]:10.3f} {p_values[i]:10.3f}"
        print(row)


    print("==================================================")

# Exécution de la fonction principale
Fonction_regression()
