import numpy as np
import pandas as pd 

def generate_readmission_column(df, readmission_col_name):
    """
    """
    result_series = [0 if x == 'NO' else 1 for x in df[readmission_col_name]]
    return result_series

df = pd.DataFrame({
    'test1': ["NO", "NO", "YES"],
    'test2': [np.nan, np.nan, np.nan]
})

def test_readmission_column():
    test_series = generate_readmission_column(df, 'test1')
    answer = pd.Series([0, 0, 1])
    pd.testing.assert_series_equal(pd.Series(test_series), answer)

def test_second_column():
    test_series = generate_readmission_column(df, 'test2')
    answer = pd.Series([1, 1, 1])
    pd.testing.assert_series_equal(pd.Series(test_series), answer)

