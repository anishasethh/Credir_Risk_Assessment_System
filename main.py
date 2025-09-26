import pandas as pd
import numpy as np  
from src.load import load_data, data_summary
from src.features import *
from src.eda import *
from src.training import *
import mlflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Credit Risk Assessment")

def main():
    app_path = 'data/application_train.csv'
    bureau_path = 'data/bureau.csv'
    df, bureau = load_data(app_path,bureau_path)
    
    # summary of data
    data_summary(df)

    # financial EDA
    # run_eda(df)

    # data cleaning and feature engineering
    df = add_bureau_features(df, bureau)
    df = newfeatures(df)
    df = clean_data(df)

    drop_cols = [
    'OWN_CAR_AGE','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL',
    'HOUR_APPR_PROCESS_START','REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION',
    'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY','ORGANIZATION_TYPE',
    'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','APARTMENTS_AVG','BASEMENTAREA_AVG','YEARS_BEGINEXPLUATATION_AVG',
    'YEARS_BUILD_AVG','COMMONAREA_AVG','ELEVATORS_AVG','ENTRANCES_AVG','FLOORSMAX_AVG','FLOORSMIN_AVG','LANDAREA_AVG',
    'LIVINGAPARTMENTS_AVG','LIVINGAREA_AVG','NONLIVINGAPARTMENTS_AVG','NONLIVINGAREA_AVG','APARTMENTS_MODE',
    'BASEMENTAREA_MODE','YEARS_BEGINEXPLUATATION_MODE','YEARS_BUILD_MODE','COMMONAREA_MODE','ELEVATORS_MODE',
    'ENTRANCES_MODE','FLOORSMAX_MODE','FLOORSMIN_MODE','LANDAREA_MODE','LIVINGAPARTMENTS_MODE','LIVINGAREA_MODE',
    'NONLIVINGAPARTMENTS_MODE','NONLIVINGAREA_MODE','APARTMENTS_MEDI','BASEMENTAREA_MEDI','YEARS_BEGINEXPLUATATION_MEDI',
    'YEARS_BUILD_MEDI','COMMONAREA_MEDI','ELEVATORS_MEDI','ENTRANCES_MEDI','FLOORSMAX_MEDI','FLOORSMIN_MEDI',
    'LANDAREA_MEDI','LIVINGAPARTMENTS_MEDI','LIVINGAREA_MEDI','NONLIVINGAPARTMENTS_MEDI','NONLIVINGAREA_MEDI',
    'FONDKAPREMONT_MODE','HOUSETYPE_MODE','TOTALAREA_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE', 'OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE',
    'OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','DAYS_LAST_PHONE_CHANGE','FLAG_DOCUMENT_2','FLAG_DOCUMENT_3',
    'FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9',
    'FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15',
    'FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21',
    'AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY',
    'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR','DAYS_BIRTH','DAYS_EMPLOYED','SK_ID_CURR','DAYS_REGISTRATION','DAYS_ID_PUBLISH']
    
    df = drop_columns(df, cols=drop_cols)

    # correlation heatmap of numerical features
    plot_corr(df)

    # spliting dataframe 
    X, y = split(df, target='TARGET')  
    
    # data preprocessing
    X_train, X_test, y_train, y_test = preprocessing(X, y)
    print("Preprocessing completed. Data is ready for model training.")

    # model training
    print("1. Logistic Regression")
    logistic_reg(X_train, X_test, y_train, y_test)

    print('2. Random Forest Classifier')
    rf(X_train, X_test, y_train, y_test)

    print('3. XGBoost Classifier')
    xgb(X_train, X_test, y_train, y_test)



if __name__ == "__main__":
    main()    
    print(">>> Credit Risk Pipeline Completed Successfully.")

