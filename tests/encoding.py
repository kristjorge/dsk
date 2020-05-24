import pandas as pd
import dsk.preprocessing as preprocessing


def main():

    le = preprocessing.LabelEncoder(method='normalized')
    dataset = pd.read_csv('../dsk/data_sets/Iris.csv')
    X = dataset.iloc[:, 1:5].values
    y = dataset.iloc[:, -1].values

    le.fit(y)
    y2 = le.transform()
    y3 = le.inverse_transform()
    a=1



if __name__ == '__main__':
    main()