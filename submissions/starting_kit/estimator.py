from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from xgboost import XGBClassifier
from sklearn.decomposition import PCA


def compute_rolling_std(X_df, feature, time_window, center=False):
    """
    For a given dataframe, compute the standard deviation over
    a defined period of time (time_window) of a defined feature

    Parameters
    ----------
    X : dataframe
    feature : str
        feature in the dataframe we wish to compute the rolling std from
    time_window : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    name = "_".join([feature, time_window, "std"])
    X_df[name] = X_df[feature].rolling(time_window, center=center).std()
    X_df[name] = X_df[name].ffill().bfill()
    X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df


def _rolling_std_dataframe(X, time_window="2H"):
    X = X.copy()
    features = X.columns
    for feature in features:
        X = compute_rolling_std(X, feature, time_window)
    return X


def compute_rolling_mean(X_df, feature, time_window, center=False):
    """
    For a given dataframe, compute the standard deviation over
    a defined period of time (time_window) of a defined feature

    Parameters
    ----------
    X : dataframe
    feature : str
        feature in the dataframe we wish to compute the rolling std from
    time_window : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    name = "_".join([feature, time_window, "mean"])
    X_df[name] = X_df[feature].rolling(time_window, center=center).mean()
    X_df[name] = X_df[name].ffill().bfill()
    X_df[name] = X_df[name].astype(X_df[feature].dtype)
    return X_df


def _rolling_mean_dataframe(X, time_window=20):
    X = X.copy()
    features = X.columns
    for feature in features:
        X = compute_rolling_mean(X, feature, time_window)
    return X

def _rolling_values_beta(X, time_windows=['1H',"4H","10H","1D"]):
    for i,  window in enumerate(time_windows):
        X = compute_rolling_mean(X, feature="Beta",time_window=window)
        X = compute_rolling_std(X, feature="Beta", time_window=window)
    return X

def _log_transformation(X):
    X = X.copy()
    proper_columns = X.columns
    columns_to_be_transformed = [colonne for colonne in X.columns if "std" not in colonne]
    columns_not_be_transformed = [colonne for colonne in X.columns if "std" in colonne]
    X_transform = X[columns_to_be_transformed]
    columns_log =X_transform.min()[(X_transform.min()>0)].index
    columns_to_be_transformed_log =X_transform.min()[(X_transform.min()<=0)].index
    X_log = np.log(X_transform[columns_log])
    X_log_bis = np.log(X_transform[columns_to_be_transformed_log]-X_transform[columns_to_be_transformed_log].min()+1)
    X_final = pd.concat([X_log, X_log_bis, X[columns_not_be_transformed]],axis=1)
    return X_final[proper_columns]


def _date_exctractor(X):
    X = X.copy()
    X.loc[:,'year'] = X.index.year
    X.loc[:, "month"] = X.index.month
    X.loc[:, "dayofweek"] = X.index.dayofweek
    X.loc[:, "week"] = X.index.isocalendar().week
    X.loc[:, "day"] = X.index.day
    X.loc[:, "hour"] = X.index.hour
    X.loc[:, "minute"] = X.index.minute
    return X

def _sinusoidale_dates(X):
    X = X.copy()  # modify a copy of X
    columns_drop = ["month", "week","day", "dayofweek", "hour", "minute"]
    periods = [
        ("month", 12),
        ("day", 31),
        ("dayofweek", 7),
        ("hour", 24),
        ("week", 52),
        ("minute", 60),
    ]
    # columns_drop = ['weekday','hour']
    # periods =[('weekday',7),('hour',24)]
    for element in periods:
        cos_col = element[0] + "_cos"
        sin_col = element[0] + "_sin"
        X.loc[:, cos_col] = X[element[0]].apply(
            lambda x: np.cos(x / element[1] * 2 * np.pi)
        )
        X.loc[:, sin_col] = X[element[0]].apply(
            lambda x: np.sin(x / element[1] * 2 * np.pi)
        )
    return X.drop(columns=columns_drop)

class FeatureExtractor(BaseEstimator):
    def fit(self, X, y):
        return self

    def transform(self, X):
        return compute_rolling_std(X, "Beta", "2h")

class CustomClassifier:

    def __init__(self, model) -> None:
        self.model = model 

    def fit(self, X, y):
        sample_weights = compute_sample_weight(
            class_weight='balanced',
            y=y #provide your own target name
        )
        try:
            self.model.fit(X,y, sample_weight=sample_weights, n_jobs=-1)
        except:
            self.model.fit(X,y, sample_weight=sample_weights)
        # self.n_outputs_ = 1
    
    def predict_proba(self, X):
        predictions = self.model.predict_proba(X)
        predictions = pd.DataFrame(predictions)
        # predictions = predictions.rolling(12, min_periods=0).quantile(0.8)
        # predictions.rolling(12, min_periods=0).quantile(0.4)
        return predictions.values

    def predict(self, X):
        proba = self.predict_proba(X)
        predictions = (proba>0.5)*1
        # ipdb.set_trace()
        return predictions

def get_estimator():
    time_window = "4H"
    std_rolling_encoder = FunctionTransformer(_rolling_std_dataframe, validate=False, kw_args={"time_window":time_window})
    mean_rolling_encoder = FunctionTransformer(_rolling_mean_dataframe, validate=False, kw_args={"time_window":time_window})
    betas_rolling_encoder = FunctionTransformer(_rolling_values_beta, validate=False, kw_args={"time_windows":['1H',"4H","10H","1D"]})
    
    date_encoder = FunctionTransformer(_date_exctractor, validate=False)
    sinusoidale_dates_encoder = FunctionTransformer(_sinusoidale_dates, validate=False)
    log_transformer = FunctionTransformer(_log_transformation, validate=False)



    transformed_dates = []
    for element in  ['month','day','day','hour', 'dayofweek','week','minute'] :
        transformed_dates.append(element+'_cos')
        transformed_dates.append(element+'_sin')
    date_cols = ['year']
    numerical_cols = ['B', 'Bx', 'Bx_rms', 'By', 'By_rms', 'Bz', 'Bz_rms', 'Na_nl', 'Np',
       'Np_nl', 'Range F 0', 'Range F 1', 'Range F 10', 'Range F 11',
       'Range F 12', 'Range F 13', 'Range F 14', 'Range F 2', 'Range F 3',
       'Range F 4', 'Range F 5', 'Range F 6', 'Range F 7', 'Range F 8',
       'Range F 9', 'V', 'Vth', 'Vx', 'Vy', 'Vz', 'Beta', 'Pdyn', 'RmsBob']
    
    std_numerical_cols = ["_".join([feature, time_window, "std"]) for feature in numerical_cols]
    mean_numerical_cols = ["_".join([feature, time_window, "mean"]) for feature in numerical_cols]


    preprocessor = ColumnTransformer([
    #    ('transformed_dates','passthrough',transformed_dates),
    #    ('dates','passthrough',date_cols),
       ("numeric",StandardScaler(), numerical_cols+std_numerical_cols+mean_numerical_cols)
    ])
    preprocessor_robust = ColumnTransformer([
       ("numeric",RobustScaler())

    ])

    classifier = RandomForestClassifier(n_estimators=100, max_depth=7,class_weight="balanced")
    classifier = XGBClassifier(**{"max_depth":10,"n_estimators":120,"learning_rate":0.001, "gamma":1,"min_child_weight":1,"verbosity":0} )
    # classifier = LogisticRegression()
    custom_classifier = CustomClassifier(classifier)
    
    classification_pipeline = Pipeline([
        ("std_encoder",std_rolling_encoder),
        ("mean_rolling_encoder", mean_rolling_encoder),
        ("betas_encoder", betas_rolling_encoder),
        # ("Log_Transformer",log_transformer),
        # ("date_encoder",date_encoder),
        # ("sinusoidales_dates",sinusoidale_dates_encoder),
        ('robust_scaler',RobustScaler()),
        ("preprocessor",StandardScaler()),
        # ("PCA", PCA(n_components=20)),
        ("model",custom_classifier)
    ])

    # feature_extractor = FeatureExtractor()

    # classifier = LogisticRegression(max_iter=1000)

    # pipe = make_pipeline(feature_extractor, StandardScaler(), classifier)
    return classification_pipeline