import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold

class PreProcessing:
    def __int__(self):
        pass

    def regexMatching(self):
        regex = "['wafer'|'Wafer']+[\_]+(\d{8}\_)+(\d{6})+\.csv"
        return regex

    def nullValueCheck(self, df):
        nullColumns = [(i, df[i].isnull().sum()) for i in df.columns if df[i].isnull().sum() > 0]
        return len(nullColumns) > 0

    def KNNImputer(self, df):
        cols = [i for i in df.columns if df[i].dtypes != 'object']
        knn = KNNImputer()
        new_df = pd.DataFrame(knn.fit_transform(df[cols]), columns=cols)
        return new_df

    def zerovarcol(self, features):
        vt = VarianceThreshold(threshold=0)
        vt.fit(features)
        zeroVarCols = [i for i in features.columns if i not in features.columns[vt.get_support()]]
        features.drop(zeroVarCols, axis=1, inplace=True)
        return features










