import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def error_df(model_name, y, pred):

    
    r2 = r2_score(y, pred)
    mse = mean_squared_error(y, pred)
    rmse = np.sqrt(mse)

    mape = np.mean(np.where(y != 0, np.abs((y - pred) / y) * 100, 0))
    mape = round(mape,2)


    df = pd.DataFrame([[model_name, r2, mse, rmse, mape]],
                        columns = ["model", "r2", "mse", "rmse", "mape"])

    return df

