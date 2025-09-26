import pandas as pd
import numpy as np

def load_data(app_path,bureau_path)-> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(app_path)
    bureau = pd.read_csv(bureau_path)
    return df, bureau

def data_summary(df:pd.DataFrame) -> pd.DataFrame:
    print("Data Shape:")
    print(df.shape)
    print("\nData Types:")
    print(df.dtypes.value_counts())
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nData Summary:")
    print(df.describe(include='all'))
    return df