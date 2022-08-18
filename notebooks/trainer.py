import importlib
import mlflow
from encoders import DistanceTransformer, TimeFeaturesEncoder
from utils import compute_rmse
from data import get_data, clean_data
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
            ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
            ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
            ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
            ])
        return pipe

    def run(self, X_train, y_train, pipeline):
        """set and train the pipeline"""
        pipeline.fit(X_train, y_train)
        return pipeline


    def evaluate(self, X_test, y_test, pipeline):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse

    @memoized_property
    def mlflow_client(self):
        MLFLOW_URI = "https://mlflow.lewagon.ai/"
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def experiment_name(self):
        EXPERIMENT_NAME = "[SG] [Singapore] [smurf] TaxiFareModel_v3"
        return EXPERIMENT_NAME

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        print(f"experiment URL: https://mlflow.lewagon.ai/#/experiments/{self.mlflow_experiment_id}")
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        return joblib.dump(self.set_pipeline , 'smhurfy_model_jlib')

if __name__ == "__main__":
    # get data
    df = get_data(nrows=10_000)

    # clean data
    clean_df = clean_data(df)

    # set X and y
    X = df
    y = df.pop("fare_amount")

    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # train
    test = Trainer(X,y)
    pipeline = test.set_pipeline()
    fitting = test.run(X_train, y_train, pipeline)

    # evaluate
    eval = test.evaluate(X_test, y_test, pipeline)

    # mlflow
    test.mlflow_log_param("student_name", "smurf")
    test.mlflow_log_metric("rmse", eval)

    save_my_model = test.save_model()

    print('done liao')
