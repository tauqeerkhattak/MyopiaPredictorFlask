import numpy as np
import pandas as pd


class Myopia:
    data_frame = pd.read_csv('https://raw.githubusercontent.com/tauqeerkhattak/dataset/main/Myopia%20Dataset.csv')
    data = np.array(data_frame.drop('Class',axis=1))
    target = np.array(data_frame['Class'])
    target_names = np.array([
        'Non Myopic',
        'Myopic',
    ])
    feature_names = np.array(data_frame.columns.drop('Class'))
