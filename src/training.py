from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import category_encoders as ce
import mlflow.sklearn
import mlflow



def split(df, target):
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()
    return X, y

# --------------------------- PREPROCESSING --------------------------------------

def preprocessing(X, y):
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

    target_enc_cols = [
        'CNT_CHILDREN','CNT_FAM_MEMBERS',
        'total_credits','active_credits','closed_credits','overdue_credits'
    ]
    for col in target_enc_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    encoder_te = ce.TargetEncoder(cols=target_enc_cols,smoothing=10,handle_unknown='value',handle_missing='value')
    X_train = encoder_te.fit_transform(X_train, y_train)
    X_test = encoder_te.transform(X_test)

    
    numeric_cols = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_GOODS_PRICE','total_overdue','total_debt','total_credit',
                    'mean_credit_duration','enddate_diff_sum','Employment_dur','Age','active/total','closed/total',
                    'overdue/total','prop_credit','prop_annuity','prop_goods_pr','credit_paid','debt_overdue']
    
    categorical_cols = ['NAME_INCOME_TYPE','NAME_FAMILY_STATUS','NAME_CONTRACT_TYPE',
                        'FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_HOUSING_TYPE','FLAG_MOBIL',
                        'OCCUPATION_TYPE','NAME_TYPE_SUITE','CODE_GENDER']
    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    # Pipelines
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    columns = ColumnTransformer([
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols),
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
         ['NAME_EDUCATION_TYPE'])
    ])

    X_train = columns.fit_transform(X_train)
    X_test = columns.transform(X_test)

    return X_train, X_test, y_train, y_test


# --------------------------------- MODELS ---------------------------------------

def logistic_reg(X_train, X_test, y_train, y_test):

    with mlflow.start_run(run_name='Logistic Regressiion'): 
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
            
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)
        y_prob = lr.predict_proba(X_test)[:, 1]

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob)}")  

        mlflow.log_param("model_type", "Logistic Regression")  
        mlflow.sklearn.log_model(lr, "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))        
    
    return lr               


def rf(X_train, X_test, y_train, y_test):
    
    with mlflow.start_run(run_name='Random Forest Classifier'):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

        rfc = RandomForestClassifier(n_estimators=100, random_state=42)
        rfc.fit(X_train, y_train)

        y_pred = rfc.predict(X_test)
        y_prob = rfc.predict_proba(X_test)[:, 1]

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob)}")

        mlflow.log_param("model_type", "Random Forest Classifier")  
        mlflow.sklearn.log_model(rfc, "RandomForestClassifier")     
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))
        
        return rfc


def xgb(X_train, X_test, y_train, y_test):

    with mlflow.start_run(run_name='XGBoost Classifier'): 
        from xgboost import XGBClassifier
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

        xgb = XGBClassifier(n_estimators=100,random_state=42)
        xgb.fit(X_train, y_train)

        y_pred = xgb.predict(X_test)
        y_prob = xgb.predict_proba(X_test)[:, 1]

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob)}")

        mlflow.log_param("model_type", "XGBoost Classifier")  
        mlflow.xgboost.log_model(xgb, "XGBClassifier")     
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob)) 
    
    return xgb



