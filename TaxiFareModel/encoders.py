from sklearn.base import BaseEstimator, TransformerMixin
from TaxiFareModel.utils import haversine_vectorized, minkowski_distance_gps, minkowski_distance, rad2dist
import pandas as pd
import numpy as np

class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
        Extracts the day of week (dow), the hour, the month and the year from a time column.
        Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'.
    """
    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_.index = pd.to_datetime(X[self.time_column])
        X_.index = X_.index.tz_convert(self.time_zone_name)
        X_["dow"] = X_.index.weekday
        X_["hour"] = X_.index.hour
        X_["month"] = X_.index.month
        X_["year"] = X_.index.year
        return X_[['dow', 'hour', 'month', 'year']]


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
        Computes the haversine distance between two GPS points.
        Returns a copy of the DataFrame X with only one column: 'distance'.
    """
    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_["distance"] = minkowski_distance_gps(X_[self.start_lat],
                                              X_[self.start_lon],
                                              X_[self.end_lat],
                                              X_[self.end_lon],1)
        return X_


class  DistanceToCenter(BaseEstimator, TransformerMixin):
    """computes distance to center of NY"""
    def __init__(self,end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        nyc_center = (40.7141667, -74.0063889)
        X_["nyc_lat"], X_["nyc_lng"] = nyc_center[0], nyc_center[1]
        args = dict(start_lat="nyc_lat",
                    start_lon="nyc_lng",
                    end_lat=self.end_lat,
                    end_lon=self.end_lon)
        X_['distance_to_center'] = haversine_vectorized(X_, **args)

        return X_


class DistanceToJFK(BaseEstimator, TransformerMixin):
    """computes distance to center of JFK"""
    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        jfk_center = (40.6441666667, -73.7822222222)

        X_["jfk_lat"], X_["jfk_lng"] = jfk_center[0], jfk_center[1]
        args_pickup =  dict(start_lat="jfk_lat", start_lon="jfk_lng",
                            end_lat="pickup_latitude", end_lon="pickup_longitude")
        args_dropoff =  dict(start_lat="jfk_lat", start_lon="jfk_lng",
                            end_lat="dropoff_latitude", end_lon="dropoff_longitude")

        X_['pickup_distance_to_jfk'] = haversine_vectorized(X_, **args_pickup)
        X_['dropoff_distance_to_jfk'] = haversine_vectorized(
            X_, **args_dropoff)

        return X_[['pickup_distance_to_jfk','dropoff_distance_to_jfk', 'distance', 'distance_to_center']]
