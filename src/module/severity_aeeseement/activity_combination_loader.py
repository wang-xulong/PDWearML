import os
import pandas as pd


class ActivityCombLoader:
    def __init__(self, back_to_root, base_path, activity_ids, combination_mode='horizontal'):

        self.back_to_root = back_to_root
        self.base_path = base_path
        self.activity_ids = activity_ids
        self.combination_mode = combination_mode
        self.data_frames = {}
        self.common_patient_ids = None
        self._load_data()

    def _load_data(self):

        for activity_id in self.activity_ids:
            file_path = os.path.join(self.back_to_root, self.base_path, f'activity_{activity_id}.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)

                if all(col in df.columns for col in ['PatientID', 'Severity_Level', 'activity_label']):

                    exclude_columns = ['PatientID', 'Severity_Level', 'activity_label']

                    feature_name = df.columns.difference(exclude_columns)

                    if self.combination_mode == 'horizontal':

                        features = df.loc[:, feature_name].add_suffix(f'_{activity_id}')
                    else:

                        features = df.loc[:, feature_name]

                    patient_ids = df['PatientID']
                    severity_levels = df['Severity_Level']


                    self.data_frames[activity_id] = (features, patient_ids, severity_levels)


                    if self.common_patient_ids is None:
                        self.common_patient_ids = set(patient_ids)
                    else:
                        self.common_patient_ids.intersection_update(patient_ids)
                else:
                    print(f"file {file_path} lack important column，skip it")
            else:
                print(f"file {file_path} no exist, skip it")
        print(f"available number of people ({len(self.common_patient_ids)}):{self.common_patient_ids}")

        self.common_patient_ids = list(self.common_patient_ids)

    def merge_weighted(self, weights):

        if not self.data_frames or not self.common_patient_ids:
            return None


        common_columns = None
        for activity_id in self.activity_ids:
            features, _, _ = self.data_frames[activity_id]
            if common_columns is None:
                common_columns = set(features.columns)
            else:
                common_columns.intersection_update(features.columns)


        if not common_columns:
            raise ValueError("没有找到共同的特征列，无法进行加权拼接。")


        common_columns = list(common_columns)


        merged_features_list = []
        patient_ids_list = []
        severity_levels_list = []


        for patient_id in self.common_patient_ids:

            max_sample_length = max(
                len(self.data_frames[activity_id][0][self.data_frames[activity_id][1] == patient_id])
                for activity_id in self.activity_ids
            )

            total_weight = 0
            patient_severity = None
            weighted_features = pd.DataFrame(0, index=range(max_sample_length), columns=common_columns)

            for activity_id, weight in zip(self.activity_ids, weights):
                features, patient_ids, severity_levels = self.data_frames[activity_id]

                patient_features = features[patient_ids == patient_id][common_columns].reset_index(drop=True)


                if len(patient_features) < max_sample_length:
                    fill_values = patient_features.mean()
                    patient_features = patient_features.reindex(range(max_sample_length), fill_value=None)
                    patient_features = patient_features.fillna(fill_values)

                weighted_features += patient_features * weight
                total_weight += weight
                patient_severity = severity_levels[patient_ids == patient_id].values[0]


            weighted_features /= total_weight

            merged_features_list.append(weighted_features)
            patient_ids_list.append(pd.Series([patient_id] * max_sample_length, name='PatientID'))
            severity_levels_list.append(pd.Series([patient_severity] * max_sample_length, name='Severity_Level'))


        if merged_features_list:
            final_merged_features = pd.concat(merged_features_list, axis=0).reset_index(drop=True)
            final_patient_ids = pd.concat(patient_ids_list, axis=0).reset_index(drop=True)
            final_severity_levels = pd.concat(severity_levels_list, axis=0).reset_index(drop=True)
        else:
            final_merged_features = pd.DataFrame()
            final_patient_ids = pd.Series(name='PatientID')
            final_severity_levels = pd.Series(name='Severity_Level')


        final_merged = pd.concat([final_merged_features, final_patient_ids, final_severity_levels], axis=1)

        return final_merged

    def merge_vertical(self):

        if not self.data_frames or not self.common_patient_ids:
            return None


        common_columns = None
        for activity_id in self.activity_ids:
            features, _, _ = self.data_frames[activity_id]
            if common_columns is None:
                common_columns = set(features.columns)
            else:
                common_columns.intersection_update(features.columns)


        if not common_columns:
            raise ValueError("没有找到共同的特征列，无法进行纵向拼接。")


        common_columns = list(common_columns)


        merged_features_list = []


        for activity_id in self.activity_ids:
            features, patient_ids, severity_levels = self.data_frames[activity_id]

            common_features = features[common_columns].copy()


            common_features.loc[:, 'PatientID'] = patient_ids.values
            common_features.loc[:, 'Severity_Level'] = severity_levels.values


            merged_features_list.append(common_features)


        final_merged = pd.concat(merged_features_list, axis=0).reset_index(drop=True)

        return final_merged

    def merge_horizontal(self):

        if not self.data_frames or not self.common_patient_ids:
            return None

        merged_features_list = []
        patient_ids_list = []
        severity_levels_list = []


        for patient_id in self.common_patient_ids:
            patient_data = []
            patient_severity = None
            for activity_id in self.activity_ids:
                features, patient_ids, severity_levels = self.data_frames[activity_id]
                patient_features = features[patient_ids == patient_id].copy()
                if patient_features.empty:
                    continue

                patient_data.append(patient_features)
                patient_severity = severity_levels[patient_ids == patient_id].values[0]
            if patient_data:
                merged_features = pd.concat(patient_data, axis=1)
                merged_features = merged_features.apply(lambda col: col.fillna(col.mean()), axis=0)
                patient_id_series = pd.Series([patient_id] * merged_features.shape[0], name='PatientID')
                patient_severity_series = pd.Series([patient_severity] * merged_features.shape[0],
                                                    name='Severity_Level')

                merged_features_list.append(merged_features)
                patient_ids_list.append(patient_id_series)
                severity_levels_list.append(patient_severity_series)

        if merged_features_list:
            final_merged_features = pd.concat(merged_features_list, axis=0).reset_index(drop=True)
            final_patient_ids = pd.concat(patient_ids_list, axis=0).reset_index(drop=True)
            final_severity_levels = pd.concat(severity_levels_list, axis=0).reset_index(drop=True)
        else:
            final_merged_features = pd.DataFrame()
            final_patient_ids = pd.Series(name='PatientID')
            final_severity_levels = pd.Series(name='Severity_Level')
        final_merged = pd.concat([final_merged_features, final_patient_ids, final_severity_levels], axis=1)

        return final_merged

    def save_merged_data(self, merged_data, output_dir):

        activity_ids_str = "_".join(map(str, self.activity_ids))
        file_name = f"merged_activities_{activity_ids_str}_{self.combination_mode}.csv"
        output_path = os.path.join(self.back_to_root, output_dir, file_name)


        merged_data.to_csv(output_path, index=False)
        print(f"文件已保存到: {output_path}")

    def combine_and_save(self, output_dir, **kwargs):
        if self.combination_mode == 'weighted':
            if kwargs['weighted'] is None:
                raise ValueError("please provide the weight values !")
            merged_data = self.merge_weighted(kwargs['weighted'])
        elif self.combination_mode == 'horizontal':
            merged_data = self.merge_horizontal()
        elif self.combination_mode == 'vertical':
            merged_data = self.merge_vertical()
        else:
            raise ValueError(f"unknown combination mode: {self.combination_mode}")

        self.save_merged_data(merged_data, output_dir)


if __name__ == '__main__':
    _back_to_root = "../../.."

    base_path = "output/feature_selection"

    activity_ids = [10, 11]

    weighted = [0.4874, 0.6808]


    # horizontal
    # vertical
    # weighted
    loader = ActivityCombLoader(_back_to_root, base_path, activity_ids, combination_mode='vertical')

    loader.combine_and_save(os.path.join("output/activity_combination"), weighted=weighted)

