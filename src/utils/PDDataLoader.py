import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from typing import Dict, List


class PDDataLoader:
    def __init__(self, activity_id: List[int], data_path: str, fold_groups_path: str,
                 severity_mapping: Dict = None, **kwargs):
        if severity_mapping is None:
            severity_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        self.data_path = data_path
        self.activity_id = activity_id
        self.severity_mapping = severity_mapping
        self.single_activity = True
        self.feature_name = None
        if len(activity_id) > 1:
            self.single_activity = False
        self.PD_data = self.load_and_preprocess_data()
        self.fold_groups = self.process_fold_groups(fold_groups_path)
        self.bag_data_dict, self.patient_ids, self.fold_groups = self.group_data(self.fold_groups)

    def process_fold_groups(self, csv_file_path):
        try:
            fold_groups_csv = pd.read_csv(csv_file_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return []
        fold_groups_csv['activity_label'] = fold_groups_csv['activity_label'].astype(str)
        if self.single_activity:
            activity_ids = str(self.activity_id[0])
        else:
            activity_ids = '+'.join(map(str, self.activity_id))
        filtered_data = fold_groups_csv.loc[fold_groups_csv['activity_label'] == activity_ids, :]
        fold_groups = []
        for i in range(1, 6):
            group_data = []
            for col_type in ['HC', 'Hy1', 'Hy2', 'Hy3', 'Hy4']:
                column_name = f'Group{i}_{col_type}'
                group_data.extend(json.loads(filtered_data.iloc[0][column_name]))
            fold_groups.append(group_data)
        return fold_groups

    def load_and_preprocess_data(self):
        data = pd.read_csv(self.data_path)
        self._validate_activity_id()
        print(f"Loading data of activity_id: {self.activity_id}")

        exclude_columns = ['PatientID', 'Severity_Level', 'activity_label']

        self.feature_name = data.columns.difference(exclude_columns)
        if self.single_activity:
            data = data.loc[data['activity_label'] == int(self.activity_id[0]), :]
            data = data.reset_index(drop=True)
        data['Severity_Level'] = data['Severity_Level'].map(self.severity_mapping)
        data = data.dropna()
        scaler = StandardScaler()
        data[self.feature_name] = scaler.fit_transform(data[self.feature_name])
        return data

    def _validate_activity_id(self):
        for a in self.activity_id:
            if not (1 <= int(a) <= 16):
                raise AssertionError(f"activity_id {a} is not in the range 1 to 16.")

    def create_train_test_split(self, fold_num, test_ids):
        train_ids = []
        for num, fold_ids in enumerate(self.fold_groups):
            if num != fold_num:
                train_ids.extend(fold_ids)
        train_X = [self.bag_data_dict[pid]["bag_data"] for pid in train_ids]
        train_X = np.vstack(train_X)
        train_Y = [self.bag_data_dict[pid]["bag_data_instance_label"] for pid in train_ids]
        train_Y = np.hstack(train_Y)

        test_X_ls = [self.bag_data_dict[pid]["bag_data"] for pid in test_ids]
        test_Y_ls = [self.bag_data_dict[pid]["bag_data_label"] for pid in test_ids]

        return train_X, train_Y, test_X_ls, test_Y_ls, train_ids, test_ids

    @staticmethod
    # 检查 self.fold_groups 中是否有重复的 ID
    def check_no_duplicates(fold_groups):
        seen_ids = set()
        for fold in fold_groups:
            for id_ in fold:
                if id_ in seen_ids:
                    raise ValueError(f"Duplicate ID found: {id_}")
                seen_ids.add(id_)
        print("No duplicate IDs found in fold_groups.")

    @staticmethod
    # 将嵌套的 fold_groups 摊平
    def flatten_fold_groups(fold_groups):
        return [id_ for fold in fold_groups for id_ in fold]

    @staticmethod
    # 检查摊平后的 fold_groups 是否全部元素属于 patient_ids
    def check_all_ids_in_patient_ids(flattened_fold_groups, patient_ids):
        patient_id_set = set(patient_ids)
        if len(patient_ids) != len(set(patient_id_set)):
            raise ValueError("patient_id_set has the same patients id")
        invalid_ids = [id_ for id_ in flattened_fold_groups if id_ not in patient_id_set]
        if invalid_ids:
            raise ValueError(f"The following IDs are not in patient_ids: {invalid_ids}")
        print("All IDs in fold_groups are in patient_ids.")

    def group_data(self, fold_groups):
        grouped = self.PD_data.groupby(['PatientID', 'Severity_Level'])
        bag_data_dict = {}
        patient_ids = []
        for (patient_id, _), group in grouped:
            self._validate_activity_id()
            bag_data = np.array(group.loc[:, self.feature_name])
            bag_data_instance_label = np.array(group['Severity_Level'])
            patient_ids.append(patient_id)
            if patient_id not in bag_data_dict:
                bag_data_dict[patient_id] = {
                    "pid": patient_id,
                    "bag_data": bag_data,
                    "bag_data_label": bag_data_instance_label[0],
                    "bag_data_instance_label": bag_data_instance_label
                }
        self.check_no_duplicates(fold_groups)
        flattened_fold_groups = self.flatten_fold_groups(fold_groups)
        self.check_all_ids_in_patient_ids(flattened_fold_groups, patient_ids)
        for fold_num, flod_group in enumerate(fold_groups):
            print(f"Group{fold_num}， {len(flod_group)}")
        print(
            f"Number of subjects：{len(flattened_fold_groups)}人, activity_id:{self.activity_id}, "
            f"Number of features：{bag_data.shape[1]}")
        return bag_data_dict, patient_ids, fold_groups


class PDDataLoaderSimple:
    def __init__(self, activity_id: List[int], data_path: str, **kwargs):
        self.data_path = data_path
        self.activity_id = activity_id
        self.single_activity = True if len(activity_id) == 1 else False
        self.exclude_columns = ['PatientID', 'Severity_Level', 'activity_label']
        self.feature_name = None
        self.valid = kwargs.get('valid', False)
        self.train_data = self.get_train_data()
        self.test_data = self.get_test_data()

    def get_train_data(self):
        train_data = pd.read_csv(os.path.join(self.data_path, 'train', f'train_activity_{self.activity_id[0]}.csv'))
        train_x = train_data.drop(columns=self.exclude_columns)
        self.feature_name = train_x.columns
        train_y = train_data['Severity_Level']
        return train_x.values, train_y.values

    def get_test_data(self):
        test_x_ls = []
        test_y_ls = []
        if self.valid:
            test_data = pd.read_csv(os.path.join(self.data_path, 'valid', f'valid_activity_{self.activity_id[0]}.csv'))
        else:
            test_data = pd.read_csv(os.path.join(self.data_path, 'test', f'test_activity_{self.activity_id[0]}.csv'))

        grouped = test_data.groupby('PatientID')


        for patient_id, group in grouped:

            test_x = group.drop(columns=self.exclude_columns)
            test_y = group['Severity_Level'].values


            test_x_ls.append(test_x.values)
            test_y_ls.append(test_y[0])

        return test_x_ls, test_y_ls



if __name__ == '__main__':
    _back_to_root = "../.."
    # doctor
    _activity_id = [11]
    _dir_path = os.path.join(_back_to_root, "output/feature_selection/doctor_label")
    simple_data = PDDataLoaderSimple(_activity_id, _dir_path, valid=True)
    print('a')
