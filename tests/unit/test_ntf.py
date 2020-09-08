from nmtf import NTF

import pandas as pd
import numpy as np
import json


def test():
    df = pd.read_csv('../data/data_Tensor.csv')
    expected_estimator = {}
    with open("../data/result.json", "r") as ifile:
        decoded_array = json.load(ifile)
        for key in decoded_array:
            expected_estimator[key] = np.asarray(decoded_array[key])
    m0 = df.values
    n_blocks = 5
    my_nt_fmodel = NTF(n_components=5)
    estimator = my_nt_fmodel.fit_transform(m0, n_blocks, sparsity=.8, n_bootstrap=10)
    estimator = my_nt_fmodel.predict(estimator)
    for key in estimator:
        assert estimator[key] == expected_estimator[key]
