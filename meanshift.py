import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == "__main__":

    dataset = pd.read_csv('./data/candy.csv')
    print(dataset.head(10))

    X = dataset.drop('competitorname', axis=1)

    """ Meanshift por defecto utiliza el ancho de banda con una tecnica matematica
    para encontrar el ancho de banda mas ajustado para nuestro modelo """
    meanshift = MeanShift().fit(X)
    print(max(meanshift.labels_))
    print("="*64)
    print(meanshift.cluster_centers_)

    dataset['meanshift'] = meanshift.labels_
    print("="*64)
    print(dataset)