import os
import sys


import warnings
warnings.filterwarnings("ignore", category=UserWarning)




# get project path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
project_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'src'))
# put src into sys.path
sys.path.append(project_root)
sys.path.append(project_src)
from src.module.feature_extraction import FeatureExtraction


# Initial parameters
back_to_root = project_root
_data_dir_path = os.path.join(back_to_root, "input/feature_extraction/raw_watch/")
_feature_file = "feature_name.csv"
_label_file = "Information_Sheet_Version_1_Watch.csv"
_patients_id = range(1, 121 + 1)
_activity_id = range(1, 16 + 1)
_side_r = "wristr"
_side_l = "wristl"
_side = "wrist"
_window_size = 150
_overlapping_rate = 0.5
_frequency = 100

# create FeatureExtraction instance
feature_extraction = FeatureExtraction(_data_dir_path, _feature_file, _label_file, _patients_id, _activity_id,
                                       _side,
                                       _window_size, _overlapping_rate, _frequency, watch=True)

# label
feature_label = feature_extraction.extract_features()

# save
output_path = r'output/feature_extraction/watch'
output_name = f"{_side}_acc_feature_label.csv"
feature_extraction.save_features(feature_label, os.path.join(back_to_root, output_path), output_name)
