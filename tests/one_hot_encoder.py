import pandas as pd
import dsk.preprocessing as preprocessing


def main():

    df = pd.read_csv('../dsk/data_sets/USA_cars_datasets.csv')
    X = df.iloc[:, [1, 2, 6]].values
    one_hot_encoder = preprocessing.OneHotEncoder()
    one_hot_encoder.fit(X, [1])
    X_v2 = one_hot_encoder.transform(X)
    X_v3 = one_hot_encoder.transform_back(X_v2)


if __name__ == '__main__':
    main()