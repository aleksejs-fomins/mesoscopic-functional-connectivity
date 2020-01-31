import numpy as np
import pandas as pd



def filter_rows_colval(df, colname, val):
    return df[df[colname] == val]


# Get rows for which several columns have some exact values
#    df.query('C1>=0 and C1<=20 and C2>=0 and C2<=20 and C3>=0 and C3<=20')
def filter_rows_colvals(df, coldict):
    # Query likes strings to be wrapped in quotation marks for later evaluation
    strwrap = lambda val: '"' + val + '"' if isinstance(val, str) else str(val)
    query = ' and '.join([colname+'=='+strwrap(val) for colname, val in coldict.items()])
    return df.query(query)


def get_one_row(rows):
    nRows = rows.shape[0]
    if nRows == 0:
        return None, None
    elif nRows > 1:
        raise ValueError("Expected 1 match, got", nRows)

    for idx, row in rows.iterrows():
        return idx, row