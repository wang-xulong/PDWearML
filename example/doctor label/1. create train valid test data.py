import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# get project path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# put src into sys.path
sys.path.append(project_root)


data_dir_path = os.path.join(project_root, "output/feature_selection/")
output_dir_path = os.path.join(project_root, "output/feature_selection/doctor_label")

train_output_path = os.path.join(output_dir_path, "train")
valid_output_path = os.path.join(output_dir_path, "valid")
test_output_path = os.path.join(output_dir_path, "test")


os.makedirs(train_output_path, exist_ok=True)
os.makedirs(valid_output_path, exist_ok=True)
os.makedirs(test_output_path, exist_ok=True)


label_path = os.path.join(project_root, "input/feature_extraction/raw/Information_Sheet_Version_1_Doctor.csv")

label_df = pd.read_csv(label_path)


def split_dataset(data_path: str, label_path: str, train_ratio: float = 0.8):

    data = pd.read_csv(data_path)
    label = pd.read_csv(label_path)


    test_label = label[label['Doctor_Level'] != -1]


    remaining_label = label[label['Doctor_Level'] == -1]


    patient_ids = remaining_label['PatientID'].unique()


    train_patient_ids, valid_patient_ids = train_test_split(patient_ids, train_size=train_ratio, random_state=42)
    test_patient_ids = test_label['PatientID'].unique()


    train_data = data[data['PatientID'].isin(train_patient_ids)]
    valid_data = data[data['PatientID'].isin(valid_patient_ids)]
    test_data = data[data['PatientID'].isin(test_patient_ids)]

    print(f"训练集大小: {train_data.shape} ")
    print(f"验证集大小: {valid_data.shape} ")
    print(f"测试集大小: {test_data.shape} ")

    return train_data, valid_data, test_data



for i in range(1, 17):

    activity_file = f"activity_{i}.csv"
    data_path = os.path.join(data_dir_path, activity_file)


    train_data, valid_data, test_data = split_dataset(data_path, label_path)


    test_data = test_data.merge(label_df[['PatientID', 'Doctor_Level']], on='PatientID', how='left')
    test_data['Severity_Level'] = test_data['Doctor_Level']
    test_data.drop(columns=['Doctor_Level'], inplace=True)


    train_data.to_csv(os.path.join(train_output_path, f"train_activity_{i}.csv"), index=False)
    valid_data.to_csv(os.path.join(valid_output_path, f"valid_activity_{i}.csv"), index=False)
    test_data.to_csv(os.path.join(test_output_path, f"test_activity_{i}.csv"), index=False)

    print(f"End of processing for activity {i}")


