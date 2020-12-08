import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from robo_adviser_sample import RoboAdviserSample

class DataLoader():
    def __init__(self, test_size = 0.2, scale = True):
        self.test_size = test_size
        self.scale = scale
        self.scaler = StandardScaler()
        self._location_map = {}
        self._sex_map = {}

    def load_preprocess(self, path):
        data = pd.read_csv(path)

        data = self._feature_engineering_pipeline(data)

        X = data.drop(['profileType', "location", "sex"], axis=1)

        if(self.scale):
            X = self.scaler.fit_transform(X) 

        y = data['profileType']
        spicies = {'Basic': 0, 'Standard': 1, 'Premium': 2}
        y = [spicies[item] for item in y]
        y = np.array(y) 

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=33)

        return X_train, X_test, y_train, y_test

    def prepare_sample(self, raw_sample: RoboAdviserSample):
        location = self._location_map[raw_sample.location]
        sex = self._sex_map[raw_sample.sex]

        sample = [raw_sample.age, raw_sample.smoker, raw_sample.familyMembers, raw_sample.salary, location, sex]
        sample = np.array([np.asarray(sample)]).reshape(-1, 1)

        if(self.scale):
            self.scaler.fit_transform(sample)

        return sample.reshape(1, -1)

    def _feature_engineering_pipeline(self, data):
        data['age'].fillna((data['age'].mean()), inplace=True)
        data['smoker'].fillna((data['smoker'].mean()), inplace=True)
        data['familyMembers'].fillna((data['familyMembers'].mean()), inplace=True)
        data['salary'].fillna((data['salary'].mean()), inplace=True)

        data["profileType"] = data["profileType"].astype('category')
        data["location"] = data["location"].astype('category')
        data["sex"] = data["sex"].astype('category')

        data["location_cat"] = data["location"].cat.codes
        data["sex_cat"] = data["sex"].cat.codes

        self._location_map = dict(zip(data['location'], data['location'].cat.codes))
        self._sex_map = dict(zip(data['sex'], data['sex'].cat.codes))

        return data
