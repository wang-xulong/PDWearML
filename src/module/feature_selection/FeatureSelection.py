import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from src.utils.PDDataLoader import PDDataLoader
from sklearn.metrics import f1_score
from src.module.feature_selection.autofeatselect import AutoFeatureSelect
from src.utils import set_seed
from typing import List


class FeatureSelection:
    def __init__(self, data_dir_path, feature_selection_dir, fold_groups_path, fold_groups_name,
                 activity_id: int, back_to_root: str, sensors: List, severity_mapping=None, seed: int = 0, ):
        self.activity_id = activity_id
        self.back_to_root = back_to_root
        self.seed = set_seed(seed)
        self.sensors = sensors
        self.data_dir_path = os.path.join(back_to_root, data_dir_path)
        self.feature_selection_dir = os.path.join(back_to_root, feature_selection_dir)
        self.fold_groups_path = os.path.join(back_to_root, fold_groups_path)
        self.fold_groups_name = fold_groups_name
        self.severity_mapping = severity_mapping
        self.data_loader = self.init_data()

    def init_data(self):
        if self.severity_mapping is None:
            self.severity_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        sensors = '_'.join(map(str, self.sensors))
        data_loader = PDDataLoader([self.activity_id], os.path.join(self.data_dir_path, sensors + "_data.csv"),
                                   os.path.join(self.fold_groups_path, self.fold_groups_name),
                                   severity_mapping=self.severity_mapping)
        return data_loader

    def identify_single_unique(self):
        data_name = f'acc_data_activity_{self.activity_id}.csv'
        data = pd.read_csv(os.path.join(self.feature_selection_dir, data_name))
        unique_counts = data.nunique()
        single_unique_features = list(unique_counts[unique_counts == 1].index)
        print(f"Identified {len(single_unique_features)} features with a single unique value.")
        return single_unique_features

    def single_activity_feature_selection(self, selection_methods):
        num_feats = [c for c in self.data_loader.feature_name]
        self.data_loader.PD_data[num_feats] = self.data_loader.PD_data[num_feats].astype('float')
        train_X, train_Y, test_X_ls, test_Y_ls, train_ids, test_ids = self.data_loader.create_train_test_split(
            0, self.data_loader.fold_groups[0])
        new_test_X = np.vstack(test_X_ls)
        new_test_Y_ls = []
        for bag_test_Y, bag_test_X in zip(test_Y_ls, test_X_ls):
            new_test_Y_ls.append(np.full(bag_test_X.shape[0], bag_test_Y))
        new_test_Y = np.hstack(new_test_Y_ls)

        train_X = pd.DataFrame(train_X, columns=num_feats)
        new_test_X = pd.DataFrame(new_test_X, columns=num_feats)
        train_Y = pd.Series(train_Y)
        new_test_Y = pd.Series(new_test_Y)

        feat_selector = AutoFeatureSelect(modeling_type='classification',
                                          X_train=train_X,
                                          y_train=train_Y,
                                          X_test=new_test_X,
                                          y_test=new_test_Y,
                                          numeric_columns=num_feats,
                                          categorical_columns=[],
                                          seed=self.seed)

        feat_selector.calculate_correlated_features(static_features=None,
                                                    num_threshold=0.9,
                                                    cat_threshold=0.9)
        feat_selector.drop_correlated_features()

        # selection_methods = ['lgbm', 'xgb', 'rf', 'perimp', 'boruta', 'rfecv']

        lgbm_hyperparams = {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 50, 'num_leaves': 10,
                            'random_state': self.seed, 'n_jobs': -1, 'importance_type': 'gain', 'verbose': -1
                            }
        rfecv_hyperparams = {'step': 3, 'n_jobs': -1, 'min_features_to_select': 3, 'cv': 5}

        final_importance_df = feat_selector.apply_feature_selection(selection_methods=selection_methods,
                                                                    lgbm_hyperparams=lgbm_hyperparams,
                                                                    xgb_hyperparams=None,
                                                                    rf_hyperparams=None,
                                                                    lassocv_hyperparams=None,
                                                                    perimp_hyperparams=None,
                                                                    rfecv_hyperparams=rfecv_hyperparams,
                                                                    boruta_hyperparams=None)

        method_to_importance_column = {
            'lgbm': 'lgbm_importance',
            'xgb': 'xgb_importance',
            'rf': 'rf_importance',
            'perimp': 'permutation_importance'
        }

        for method in selection_methods:
            if method in method_to_importance_column:
                importance_col = method_to_importance_column[method]
                rank_col_name = importance_col.replace('importance', 'ranking')
                final_importance_df[rank_col_name] = final_importance_df[importance_col].rank(ascending=False)

        file_path = os.path.join(self.feature_selection_dir,
                                 f'feature_selection_results_activity_{self.activity_id}.csv')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        final_importance_df.to_csv(file_path, index=False)

    def single_activity_best_num_features(self):
        # using lgbm as the model
        ranking_importance_path = os.path.join(self.feature_selection_dir,
                                               f'feature_selection_results_activity_{self.activity_id}.csv')
        ranking_df = pd.read_csv(ranking_importance_path)
        ranking_columns = [col for col in ranking_df.columns if 'ranking' in col.lower() or 'rankings' in col.lower()]

        ranking_df['average_ranking'] = ranking_df[ranking_columns].mean(axis=1)
        sorted_features = ranking_df.sort_values(by='average_ranking').index
        results_df = pd.DataFrame(
            columns=['num_features', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5', 'mean_score'])

        params = {
            'boosting': 'gbdt',
            'objective': 'multiclass',
            'metric': 'multi_error',
            'num_class': len(set(self.data_loader.severity_mapping.values())),
            'max_depth': 7,
            'seed': self.seed,
            'verbose': -1,
        }

        for n in range(1, len(sorted_features) + 1):
            print(f'Using {n} feature(s)')
            selected_features = sorted_features[:n]
            f1_scores = []

            for fold_num, test_ids in enumerate(self.data_loader.fold_groups):
                train_X, train_Y, test_X_ls, test_Y_ls, train_ids, test_ids = self.data_loader.create_train_test_split(
                    fold_num, test_ids)

                train_X = train_X[:, selected_features]
                test_X_ls = [test_X[:, selected_features] for test_X in test_X_ls]

                lgb_train = lgb.Dataset(train_X, train_Y)
                new_test_X = np.vstack(test_X_ls)
                new_test_Y_ls = []
                for bag_test_Y, bag_test_X in zip(test_Y_ls, test_X_ls):
                    new_test_Y_ls.append(np.full(bag_test_X.shape[0], bag_test_Y))
                new_test_Y = np.hstack(new_test_Y_ls)
                lgb_test = lgb.Dataset(new_test_X, new_test_Y, reference=lgb_train)

                model = lgb.train(
                    params,
                    lgb_train,
                    valid_sets=[lgb_train, lgb_test],
                    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
                )
                y_pred = model.predict(new_test_X, num_iteration=model.best_iteration)
                y_pred_labels = y_pred.argmax(axis=1)
                f1 = f1_score(new_test_Y, y_pred_labels, zero_division=0, average='macro')
                f1_scores.append(f1)

            mean_f1_score = np.mean(f1_scores)
            print('mean_f1_score ', mean_f1_score)

            result_row = pd.DataFrame([{
                'num_features': n,
                'fold_1': f1_scores[0],
                'fold_2': f1_scores[1],
                'fold_3': f1_scores[2],
                'fold_4': f1_scores[3],
                'fold_5': f1_scores[4],
                'mean_score': mean_f1_score
            }])
            results_df = pd.concat([results_df, result_row], ignore_index=True)

        if not results_df.empty:
            best_num_features = results_df.loc[results_df['mean_score'].idxmax(), 'num_features']
            print(f'best number of features: {best_num_features}')
            print(results_df)
            results_df.to_csv(
                os.path.join(self.feature_selection_dir, f'best_num_features_activity_{self.activity_id}.csv'),
                index=False)
        else:
            print("No valid results were obtained. Please check your data and parameters.")

    def important_feature_columns(self):  # identify the importance feature_columns
        ranking_importance_path = os.path.join(self.feature_selection_dir,
                                               f'feature_selection_results_activity_{self.activity_id}.csv')
        ranking_df = pd.read_csv(ranking_importance_path)
        ranking_columns = [col for col in ranking_df.columns if 'ranking' in col.lower() or 'rankings' in col.lower()]

        ranking_df['average_ranking'] = ranking_df[ranking_columns].mean(axis=1)
        sorted_features = ranking_df.sort_values(by='average_ranking')
        unique_columns = ['acc_a_max_a', 'acc_a_max_x', 'acc_a_max_y', 'acc_a_max_z', 'acc_a_mean_a', 'acc_a_mean_x',
                          'acc_a_mean_y', 'acc_a_mean_z']
        remaining_values = [value for value in sorted_features['feature'] if value not in unique_columns]
        return remaining_values

    def save_important_feature(self):
        # feature_path = self.feature_selection_dir
        # feature_name = f'feature_selection_results_activity_{self.activity_id}.csv'
        # feature = pd.read_csv(os.path.join(feature_path, feature_name))
        # data = pd.read_csv(os.path.join(self.data_dir_path, "acc_data.csv"))
        feature_column = self.important_feature_columns()
        label_info = ['PatientID', 'activity_label', 'Severity_Level']
        data = self.data_loader.PD_data.loc[:, feature_column + label_info]
        activity_id_filtered_data = data[data['activity_label'] == self.activity_id]
        file = os.path.join(self.feature_selection_dir, f'activity_{self.activity_id}.csv')
        activity_id_filtered_data.to_csv(file, index=False)


if __name__ == '__main__':
    back_to_root = "../../.."

    for i in range(1, 16 + 1):
        # instance feature selection
        fs = FeatureSelection(data_dir_path="output/select_sensors",
                              feature_selection_dir="output/feature_selection",
                              fold_groups_path="input/feature_extraction",
                              fold_groups_name="fold_groups_new_with_combinations.csv",
                              activity_id=i, back_to_root=back_to_root, sensors=['acc'], seed=0
                              )

        # single activity feature selection
        selection_methods = ['lgbm', 'xgb', 'rf', 'perimp', 'boruta', 'rfecv']
        fs.single_activity_feature_selection(selection_methods)

        # Analysis the performance for different num of features
        fs.single_activity_best_num_features()

        # save activity data
        fs.save_important_feature()
        print(f"Saved activity data {i}")
