import os
import pandas as pd
from src.module.feature_extraction.feature_utils import pd_utils, utils_watch


class FeatureExtraction:
    def __init__(self, data_dir_path, feature_file, label_file, patients_id, activity_id, side,
                 window_size=300, overlapping_rate=0.5,
                 frequency=200, watch=False):
        self.data_dir_path = data_dir_path
        self.feature_file = feature_file
        self.label_file = label_file
        self.patients_id = patients_id
        self.activity_id = activity_id
        self.side = side
        self.window_size = window_size
        self.overlapping_rate = overlapping_rate
        self.frequency = frequency
        self.watch = watch
        self.fea_column = self._load_feature_names()

    def _load_feature_names(self):


        feature_name_path = os.path.join(self.data_dir_path, self.feature_file)
        df_read = pd.read_csv(feature_name_path)


        acc_list = df_read['acc'].tolist()
        fea_column = acc_list
        gyro_list = df_read['gyro'].tolist()
        if self.watch is False:
            mag_list = df_read['mag'].tolist()
            fea_column += gyro_list + mag_list

        return fea_column

    def extract_features(self):

        if self.watch is False:
            Feature = pd_utils.FeatureExtractWithProcess(self.patients_id, self.activity_id, self.data_dir_path,
                                                         self.side,
                                                         self.fea_column, self.window_size, self.overlapping_rate,
                                                         self.frequency)
        else:
            sensor = "acc"
            Feature = utils_watch.FeatureExtractWithProcess1(self.patients_id, self.activity_id, sensor,
                                                          self.data_dir_path, self.side,
                                                          self.fea_column, self.window_size, self.overlapping_rate,
                                                          self.frequency)


        label_data = pd.read_csv(os.path.join(self.data_dir_path, self.label_file))
        label_table = label_data.loc[:, ["PatientID", "Severity_Level"]]


        Feature['PatientID'] = Feature['PatientID'].astype(int)
        label_table['PatientID'] = label_table['PatientID'].astype(int)


        feature_label = pd.merge(Feature, label_table, on='PatientID')

        return feature_label

    def save_features(self, feature_label, output_path, output_name):

        output_file_path = os.path.join(output_path, output_name)
        feature_label.to_csv(output_file_path, index=False)
        print(f"save the feature and corresponding labels to : {output_file_path}")


if __name__ == '__main__':

    back_to_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

    _data_dir_path = os.path.join(back_to_root, "input/feature_extraction/raw/")
    _feature_file = "feature_name.csv"
    _label_file = "Information_Sheet_Version_1.csv"
    _patients_id = range(15, 135 + 1)
    _activity_id = range(1, 16 + 1)
    _side_r = "wristr"
    _side_l = "wristl"
    _side = "wrist"
    _window_size = 300
    _overlapping_rate = 0.5
    _frequency = 200


    feature_extraction = FeatureExtraction(_data_dir_path, _feature_file, _label_file, _patients_id, _activity_id,
                                           _side_r,
                                           _window_size, _overlapping_rate, _frequency)


    feature_label = feature_extraction.extract_features()


    _output_path = r'output/feature_extraction/'
    _output_name = f"{_side_r}_acc_gyro_mag_feature_label.csv"
    feature_extraction.save_features(feature_label, os.path.join(back_to_root, _output_path), _output_name)
