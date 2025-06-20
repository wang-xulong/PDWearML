import os
import sys

# get project path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
project_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'src'))
# put src into sys.path
sys.path.append(project_root)
sys.path.append(project_src)
from src.module.select_sensors import select_data

back_to_root = project_root
data_dir_path = 'output/feature_extraction/watch'
data_file_name = 'wrist_acc_feature_label.csv'
feature_name_dir_path = 'input/feature_extraction/raw_watch/'
feature_name_file_name = 'feature_name.csv'
output_path = 'output/select_sensors/watch'

select_data(
    column_names=['acc'],
    back_to_root=back_to_root,
    data_dir_path=os.path.join(back_to_root, data_dir_path),
    data_file_name=data_file_name,
    feature_name_dir_path=os.path.join(back_to_root, feature_name_dir_path),
    feature_name_file_name=feature_name_file_name,
    output_path=output_path
)

