from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from TaxiFareModel.encoders import DistanceToCenter, DistanceTransformer,DistanceToJFK
from TaxiFareModel.encoders import TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import clean_data, get_data
from memoized_property import memoized_property
import mlflow
from  mlflow.tracking import MlflowClient
import joblib
import xgboost as xgb


EXPERIMENT_NAME = "[PT] [Lisbon] [J93s] Taxi_model 1.0"
MLFLOW_URI = "https://mlflow.lewagon.co/"

class Trainer():
    """Please enter training data and model type"""
    def __init__(self, X_train, y_train, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X_train = X_train
        self.y_train = y_train
        self.experiment_name = EXPERIMENT_NAME
        if "regressor" in kwargs.keys():
            self.regressor = kwargs["regressor"]
        else:
            self.regressor= LinearRegression()

        if self.regressor == "linear_model":
            self.model = LinearRegression()
        elif self.regressor == "random_forest":
            self.model = RandomForestRegressor()
        elif self.regressor == "XGB":
            self.model = xgb.XGBRegressor()

#,('dist_trans', DistanceTransformer())
#('dist_JFK',DistanceToJFK()),
    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([('dist_cent', DistanceToCenter()),
                              ('dist',DistanceTransformer()),
                                ('dist_JFK',DistanceToJFK()),
                          ('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                          ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
        "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
        'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                     remainder="drop")
        pipe = Pipeline([('preproc', preproc_pipe),
                     ('linear_model', self.model),
                     ])
        self.mlflow_log_param('model', self.regressor)
        return pipe

    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        self.mlflow_log_metric('RMSE',rmse)
        return rmse

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get data
    data = get_data()
    # clean data
    data = clean_data(data)
    # set X and y
    y = data["fare_amount"]
    X = data.drop("fare_amount", axis=1)
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    # train
    model_list = ["linear_model", "random_forest", "XGB"]
    for i in model_list:
        trainer = Trainer(X_train, y_train, regressor=i)
        print(i)
        # trainer.mlflow_client()
        # trainer.mlflow_experiment_id()
        # trainer.mlflow_run()
        trainer.run()
        # evaluate
        trainer.evaluate(X_test, y_test)
        trainer.save_model()
        print('TODO')
