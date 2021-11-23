from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from TaxiFareModel.encoders import DistanceTransformer
from TaxiFareModel.encoders import TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import clean_data, get_data


class Trainer():
    def __init__(self, X_train, y_train):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X_train = X_train
        self.y_train = y_train

    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                          ('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                          ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
        "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
        'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                     remainder="drop")
        pipe = Pipeline([('preproc', preproc_pipe),
                     ('linear_model', LinearRegression())])
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
        return rmse


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
    trainer = Trainer(X_train, y_train)
    trainer.run()
    # evaluate
    trainer.evaluate(X_test, y_test)
    print('TODO')
