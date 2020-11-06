import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/heart.csv')

    print(dt_heart.head(5))

    # con drop borramos valores en especifico en un dataframe y le pones como primer
    # parametro la lista de nombres de la columna y con axis=1 indicamos que se
    # borraran todos los datos de la columna.
    dt_features = dt_heart.drop(['target'], axis=1)

    # Asignamos nuestro target por separado
    dt_target = dt_heart['target']

    dt_features = StandardScaler().fit_transform(dt_features)

    X_train, X_test, y_train, y_test = train_test_split(
        dt_features, dt_target, test_size=0.3, random_state=42
    )
    
    """ Como parametro en el kernel se pueden pasar 3 valores mas usados
    los cuales son linear(lineal), poly(Polinomial) y rbf(Gaussiano) """
    kpca = KernelPCA(n_components=4, kernel='poly')
    kpca.fit(X_train)

    dt_train = kpca.transform(X_train)
    dt_test = kpca.transform(X_test)

    logistic = LogisticRegression(solver='lbfgs')

    logistic.fit(dt_train, y_train)
    print("Score KPCA: ", logistic.score(dt_test, y_test))