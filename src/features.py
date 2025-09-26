import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def clean_data(df:pd.DataFrame) -> pd.DataFrame:
    placeholders = ["XNA", "XAP", "Unknown", "NA", "N/A", "0"]
    df = df.replace(placeholders, np.nan)
    df['NAME_TYPE_SUITE'] = df['NAME_TYPE_SUITE'].replace([0, 'Other_A', 'Other_B'], np.nan)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].mean())
    return df


def newfeatures(df: pd.DataFrame) -> pd.DataFrame:
    if 'DAYS_BIRTH' in df.columns:
        df['Age'] = (df['DAYS_BIRTH'] / -365).round().astype(int)
    if 'DAYS_EMPLOYED' in df.columns:
        df['Employment_dur'] = (df['DAYS_EMPLOYED'] / -365).round().astype('Int64')
    if {'AMT_CREDIT', 'AMT_ANNUITY','AMT_GOODS_PRICE','AMT_INCOME_TOTAL'}.issubset(df.columns):
        df['loan_dur'] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"]+1)
        df['prop_credit'] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
        df['prop_goods_pr'] = df["AMT_GOODS_PRICE"] / df["AMT_INCOME_TOTAL"]
        df['prop_annuity'] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    if {'active_credits', 'closed_credits','total_credits','overdue_credits'}.issubset(df.columns):
        df["active/total"]  = df["active_credits"]  / df["total_credits"]
        df["closed/total"]  = df["closed_credits"]  / df["total_credits"]
        df["overdue/total"] = df["overdue_credits"] / df["total_credits"]
    if {'total_credit', 'total_debt','total_overdue'}.issubset(df.columns):
        df["credit_paid"]   = df["total_credit"] - df["total_debt"]
        df["debt_overdue"]  = df["total_debt"] - df["total_overdue"]
    return df


def add_bureau_features(df: pd.DataFrame, bureau: pd.DataFrame) -> pd.DataFrame:
    bureau_copy = bureau.copy()
    
    required_cols = ['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE',
                     'DAYS_ENDDATE_FACT', 'CREDIT_ACTIVE', 'AMT_CREDIT_SUM_OVERDUE',
                     'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM']
    
    missing = [c for c in required_cols if c not in bureau.columns]
    if missing:
        raise ValueError(f"Missing columns in bureau: {missing}")
    
    bureau_copy['CREDIT_DURATION'] = bureau_copy['DAYS_CREDIT_ENDDATE'] - bureau_copy['DAYS_CREDIT']
    bureau_copy['ENDDATE_DIFF'] = bureau_copy['DAYS_ENDDATE_FACT'] - bureau_copy['DAYS_CREDIT_ENDDATE']
    bureau_copy['IS_ACTIVE'] = bureau_copy['CREDIT_ACTIVE'].apply(lambda x: 1 if x == 'Active' else 0)
    bureau_copy['IS_CLOSED'] = bureau_copy['CREDIT_ACTIVE'].apply(lambda x: 1 if x == 'Closed' else 0)
    bureau_copy['HAS_OVERDUE'] = bureau_copy['AMT_CREDIT_SUM_OVERDUE'].apply(lambda x: 1 if x > 0 else 0)

    bureau_agg = bureau_copy.groupby('SK_ID_CURR').agg(
        total_credits=('SK_ID_BUREAU', 'count'),
        active_credits=('IS_ACTIVE', 'sum'),
        closed_credits=('IS_CLOSED', 'sum'),
        overdue_credits=('HAS_OVERDUE', 'sum'),
        total_overdue=('AMT_CREDIT_SUM_OVERDUE', 'sum'),
        total_debt=('AMT_CREDIT_SUM_DEBT', 'sum'),
        total_credit=('AMT_CREDIT_SUM', 'sum'),
        mean_credit_duration=('CREDIT_DURATION', 'mean'),
        enddate_diff_sum=('ENDDATE_DIFF', 'sum'),
    )

    df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
    df.fillna(0, inplace=True)
    return df


def drop_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    return df.drop(columns=[c for c in cols if c in df.columns])


def plot_corr(df: pd.DataFrame) -> None:
    """Show correlation heatmap of numerical features."""
    df_num = df.select_dtypes(include=['int64', 'float64'])
    corr_matrix = df_num.corr()

    plt.figure(figsize=(16, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.show()
