import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, \
    f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import warnings
from imblearn.over_sampling import RandomOverSampler
import shap
import yaml
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from src.utils.PDDataLoader import PDDataLoader

warnings.filterwarnings(
    "ignore", category=UserWarning, module="lightgbm.engine", lineno=172
)


def map_activity_ids(activity_ids):
    classifier_mapping = {
        1: 'FT', 2: 'FOC', 3: 'PSM', 4: 'RHF', 5: 'LHF', 6: 'FN-L', 7: 'FN-R', 8: 'FRA', 9: 'WALK', 10: 'AFC',
        11: 'DRINK', 12: 'PICK', 13: 'SIT', 14: 'STAND', 15: 'SWING', 16: 'DRAW'
    }

    mapped_values = [classifier_mapping.get(id_, 'Unknown') for id_ in activity_ids]

    if len(mapped_values) == 1:
        return mapped_values[0]

    return f"[{', '.join(mapped_values)}]"


# seed = set_seed(0)


class ModelTrainer:
    def __init__(self, classifier, params):
        self.classifier = classifier
        self.params = params
        self.model = None
        self.create_model()

    def create_model(self):
        if self.classifier == 'logistic_l1':
            self.model = make_pipeline(StandardScaler(), LogisticRegression(**self.params))
        elif self.classifier == 'logistic_l2':
            self.model = make_pipeline(StandardScaler(), LogisticRegression(**self.params))
        elif self.classifier == 'svm_l1':
            self.model = make_pipeline(StandardScaler(), LinearSVC(**self.params))
        elif self.classifier == 'svm_l2':
            self.model = make_pipeline(StandardScaler(), LinearSVC(**self.params))
        elif self.classifier == 'knn':
            self.model = make_pipeline(StandardScaler(), KNeighborsClassifier(**self.params))
        elif self.classifier == 'bayes':
            self.model = make_pipeline(StandardScaler(), GaussianNB(**self.params))
        elif self.classifier == 'rf':
            self.model = make_pipeline(StandardScaler(), RandomForestClassifier(**self.params))
        elif self.classifier == 'lgbm':
            pass
        elif self.classifier == 'xgb':
            pass
        elif self.classifier == 'mlp_2':
            self.params['hidden_layer_sizes'] = (128, 64)
            self.params['shuffle'] = False
            self.model = make_pipeline(StandardScaler(), MLPClassifier(**self.params))
        elif self.classifier == 'mlp_4':
            self.params['hidden_layer_sizes'] = (64, 32, 16, 8)
            self.model = make_pipeline(StandardScaler(), MLPClassifier(**self.params))
        elif self.classifier == 'mlp_8':
            self.params['hidden_layer_sizes'] = (256, 128, 64, 64, 32, 32, 16, 8)
            self.model = make_pipeline(StandardScaler(), MLPClassifier(**self.params))
        else:
            raise ValueError("Unsupported classifier type.")


class ModelEvaluator:
    def __init__(self, data_loader, model_trainer, **kwargs):
        self.data_loader = data_loader
        self.model_trainer = model_trainer
        self.activity_id = data_loader.activity_id
        self.classifier = model_trainer.classifier
        self.model = model_trainer.model
        self.test_X = list()
        self.feature_name = data_loader.feature_name
        self.roc = kwargs.get('roc', False)
        self.roc_result = None
        self.roc_class_result = None
        print("Training...")

    def train_evaluate(self, callback=None):
        total_pred_group_ls = []
        total_pred_proba_group_ls = []
        total_test_Y_group_ls = []
        id_records_for_each_fold_dict = dict()
        for fold_num, test_ids in enumerate(self.data_loader.fold_groups):
            # 创建一个 RandomOverSampler 实例以验证安装成功
            ros = RandomOverSampler(random_state=0)
            train_X, train_Y, test_X_ls, test_Y_ls, train_ids, test_ids = self.data_loader.create_train_test_split(
                fold_num, test_ids)
            train_X, train_Y = ros.fit_resample(train_X, train_Y)
            if fold_num not in id_records_for_each_fold_dict:
                id_records_for_each_fold_dict[fold_num] = {'train_ids': train_ids, 'test_ids': test_ids}

            if self.classifier == 'lgbm':
                dataset_scaler = StandardScaler()
                train_X = dataset_scaler.fit_transform(train_X)
                for idx in range(len(test_X_ls)):
                    test_X_ls[idx] = dataset_scaler.transform(test_X_ls[idx])
                lgb_train = lgb.Dataset(train_X, train_Y)
                new_test_Y_ls = [np.full(bag_test_X.shape[0], bag_test_Y) for bag_test_Y, bag_test_X in
                                 zip(test_Y_ls, test_X_ls)]
                new_test_X = np.vstack(test_X_ls)
                new_test_Y = np.hstack(new_test_Y_ls)
                self.test_X.append(new_test_X)
                lgb_test = lgb.Dataset(new_test_X, new_test_Y, reference=lgb_train)
                self.model = lgb.train(
                    self.model_trainer.params,
                    lgb_train,
                    num_boost_round=500,
                    valid_sets=[lgb_train, lgb_test],
                    callbacks=[lgb.early_stopping(stopping_rounds=50)]
                )
                y_pred = self.model.predict(new_test_X, num_iteration=self.model.best_iteration)
                y_pred_labels = y_pred.argmax(axis=1)
                f1 = f1_score(new_test_Y, y_pred_labels, zero_division=0, average='macro')
                print(f1)
            elif self.classifier == 'xgb':
                dataset_scaler = StandardScaler()
                train_X = dataset_scaler.fit_transform(train_X)
                for idx in range(len(test_X_ls)):
                    test_X_ls[idx] = dataset_scaler.transform(test_X_ls[idx])
                xgb_train = xgb.DMatrix(train_X, label=train_Y)
                xgb_test_X_ls = [xgb.DMatrix(test_X, label=np.full(test_X.shape[0], test_Y)) for test_X, test_Y in
                                 zip(test_X_ls, test_Y_ls)]
                evallist = [(xgb_train, 'train'), (xgb_test_X_ls[0], 'test')]
                self.model_trainer.params['random_state'] = 0
                self.model_trainer.params['nthread'] = 1
                self.model_trainer.params['tree_method'] = 'approx'
                self.model = xgb.train(self.model_trainer.params, xgb_train,
                                       evals=evallist,
                                       early_stopping_rounds=50, verbose_eval=False)
                test_X_ls = xgb_test_X_ls
            else:
                self.model.fit(train_X, train_Y)
                self.test_X.append(np.vstack(test_X_ls))

            y_pred_ls, y_pred_ls_proba = self.predict_most_likely_class(test_X_ls)
            # 将当前留一验证的病人级别的预测标签和真实标签记录
            total_pred_group_ls.append(y_pred_ls)
            total_pred_proba_group_ls.append(y_pred_ls_proba)
            total_test_Y_group_ls.append(test_Y_ls)

        # 指标估计
        metrics = ModelEvaluator._metrics_calculation(total_test_Y_group_ls, total_pred_group_ls)
        if self.roc is True:
            total_fpr_micro, total_tpr_micro, total_roc_auc_micro = (ModelEvaluator.
                                                                     _calculate_roc_curve(total_pred_proba_group_ls,
                                                                                          total_test_Y_group_ls))
            self.roc_result = (total_fpr_micro, total_tpr_micro, total_roc_auc_micro)
            total_fpr, total_tpr, total_roc_auc = (ModelEvaluator._calculate_roc_curve_class
                                                   (total_pred_proba_group_ls,
                                                    total_test_Y_group_ls))
            self.roc_class_result = (total_fpr, total_tpr, total_roc_auc)

        # 打印返回的各个值
        print("Mean Accuracy:", metrics['mean_accuracy'])
        print("Mean Precision:", metrics['mean_precision'])
        print("Mean Recall:", metrics['mean_recall'])
        print("Mean F1 Score:", metrics['mean_f1'])
        print("Mean Specificity:", metrics['mean_specificity'])
        print("Mean Report:\n", metrics['mean_report'])
        return metrics

    def predict_most_likely_class(self, test_X_ls):
        y_pred_ls = []
        y_pred_ls_proba = []
        n_classes = len(set(self.data_loader.PD_data['Severity_Level']))
        for test_X in test_X_ls:
            if self.classifier in ['logistic_l1', 'logistic_l2', 'knn', 'bayes', 'rf', 'mlp_2', 'mlp_4', 'mlp_8']:
                y_pred_prob = self.model.predict_proba(test_X)
                y_pred = np.argmax(y_pred_prob, axis=1)
            elif self.classifier in ['svm_l1', 'svm_l2']:
                y_pred = self.model.predict(test_X)
            elif self.classifier == 'lgbm':
                y_pred_prob = self.model.predict(test_X, num_iteration=self.model.best_iteration)
                y_pred = np.argmax(y_pred_prob, axis=1)
            elif self.classifier == 'xgb':
                y_pred_prob = self.model.predict(test_X, iteration_range=(0, self.model.best_iteration))
                y_pred = np.argmax(y_pred_prob, axis=1)
            else:
                raise ValueError("Unsupported classifier type for prediction.")
            counts = np.bincount(y_pred)
            # 如果 counts 的长度不足 n_classes，补齐为 0
            if len(counts) < n_classes:
                counts = np.pad(counts, (0, n_classes - len(counts)), 'constant')
            # 计算总计数
            total = np.sum(counts)
            # 计算每个类别的概率
            probabilities = counts / total
            y_pred_ls.append(np.argmax(counts))
            y_pred_ls_proba.append(probabilities)
        return y_pred_ls, y_pred_ls_proba

    @staticmethod
    def _calculate_roc_curve_class(total_pred_proba_group_ls, total_test_Y_group_ls):
        total_fpr = []
        total_tpr = []
        total_roc_auc = []

        for fold_num, (y_pred_proba, y_true) in enumerate(zip(total_pred_proba_group_ls, total_test_Y_group_ls)):
            n_classes = len(y_pred_proba[0])
            y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
            y_true_bin = np.array(y_true_bin)
            y_pred_proba = np.array(y_pred_proba)
            # 计算每个类的 ROC 曲线和 AUC
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            total_fpr.append(fpr)
            total_tpr.append(tpr)
            total_roc_auc.append(roc_auc)

        return total_fpr, total_tpr, total_roc_auc

    @staticmethod
    def _calculate_roc_curve(total_pred_proba_group_ls, total_test_Y_group_ls):
        print("calculate_roc_curve")
        total_fpr_micro = []
        total_tpr_micro = []
        total_roc_auc_micro = []

        for fold_num, (y_pred_proba, y_true) in enumerate(zip(total_pred_proba_group_ls, total_test_Y_group_ls)):
            n_classes = len(y_pred_proba[0])
            y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
            # 计算微平均 ROC 曲线和 AUC
            y_true_bin = np.array(y_true_bin)
            y_pred_proba = np.array(y_pred_proba)

            fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
            roc_auc_micro = auc(fpr_micro, tpr_micro)

            total_fpr_micro.append(fpr_micro)
            total_tpr_micro.append(tpr_micro)
            total_roc_auc_micro.append(roc_auc_micro)

        return total_fpr_micro, total_tpr_micro, total_roc_auc_micro

    @staticmethod
    def _calculate_specificity(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn = cm[0, 0]
        fp = cm[0, 1]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return specificity

    @staticmethod
    def _check_confusion_matrix_size(total_confusion_matrix_ls):
        num_classes = 4
        adjusted_total_confusion_matrix_ls = []
        for cm in total_confusion_matrix_ls:
            if cm.shape != (num_classes, num_classes):

                new_cm = np.zeros((num_classes, num_classes))

                min_dim = min(cm.shape[0], num_classes)
                new_cm[:min_dim, :min_dim] = cm[:min_dim, :min_dim]
                adjusted_total_confusion_matrix_ls.append(new_cm)
            else:
                adjusted_total_confusion_matrix_ls.append(cm)
        return adjusted_total_confusion_matrix_ls

    @staticmethod
    def _metrics_calculation(total_test_Y_group_ls, total_pred_group_ls):
        print("Evaluating...")
        total_acc_ls = []
        total_precision_ls = []
        total_recall_ls = []
        total_f1_ls = []
        total_specificity_ls = []
        report_ls = []
        total_confusion_matrix_ls = []  # To save the confusion matrix on each fold
        fold_metrics = []  # 用于存储每个 fold 的指标

        for fold_num, (total_test_Y, total_pred) in enumerate(zip(total_test_Y_group_ls, total_pred_group_ls)):
            total_acc = accuracy_score(total_test_Y, total_pred)
            total_precision = precision_score(total_test_Y, total_pred, zero_division=0, average='macro')
            total_recall = recall_score(total_test_Y, total_pred, zero_division=0, average='macro')
            total_f1 = f1_score(total_test_Y, total_pred, zero_division=0, average='macro')
            report = classification_report(total_test_Y, total_pred, zero_division=0)
            total_confusion_matrix = confusion_matrix(total_test_Y, total_pred, normalize='true')

            # 调用 calculate_specificity 静态方法计算特异性
            specificity = ModelEvaluator._calculate_specificity(total_test_Y, total_pred)

            total_acc_ls.append(total_acc)
            total_precision_ls.append(total_precision)
            total_recall_ls.append(total_recall)
            total_f1_ls.append(total_f1)
            total_specificity_ls.append(specificity)
            report_ls.append(report)
            total_confusion_matrix_ls.append(total_confusion_matrix)

            # 记录每个 fold 的指标
            fold_metrics.append({
                'fold_num': fold_num + 1,  # fold 编号，从 1 开始
                'accuracy': total_acc,
                'precision': total_precision,
                'recall': total_recall,
                'f1': total_f1,
                'specificity': specificity,
                'report': report
            })

        mean_acc = round(np.mean(total_acc_ls), 4)
        mean_precision = round(np.mean(total_precision_ls), 4)
        mean_recall = round(np.mean(total_recall_ls), 4)
        mean_f1 = round(np.mean(total_f1_ls), 4)
        mean_specificity = round(np.mean(total_specificity_ls), 4)
        all_total_test_Y = [item for sublist in total_test_Y_group_ls for item in sublist]
        all_total_pred = [item for sublist in total_pred_group_ls for item in sublist]
        all_report = classification_report(all_total_test_Y, all_total_pred, zero_division=0)
        # Calculate the mean normalized confusion matrix across all folds
        total_confusion_matrix_ls = ModelEvaluator._check_confusion_matrix_size(total_confusion_matrix_ls)
        mean_confusion_matrix = np.round(np.mean(np.array(total_confusion_matrix_ls), axis=0), 2)

        return {
            'mean_accuracy': mean_acc,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_f1': mean_f1,
            'mean_specificity': mean_specificity,
            'mean_report': all_report,
            'fold_metrics': fold_metrics,  # 返回每个 fold 的指标
            'mean_confusion_matrix': mean_confusion_matrix
        }


def load_config(activity_id: int):
    config_path = f'config/activity_{activity_id}.yaml'
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


class ModelEvaluatorSimple:
    def __init__(self, data_loader, model_trainer, **kwargs):
        self.data_loader = data_loader
        self.model_trainer = model_trainer
        self.activity_id = data_loader.activity_id
        self.classifier = model_trainer.classifier
        self.model = model_trainer.model
        self.test_X = list()
        self.feature_name = data_loader.feature_name
        self.roc = kwargs.get('roc', False)
        self.roc_result = None
        self.roc_class_result = None
        print("Training...")

    def train_evaluate(self):
        ros = RandomOverSampler(random_state=0)
        train_x, train_y = self.data_loader.train_data
        test_x_ls, test_y_ls = self.data_loader.test_data
        train_x, train_y = ros.fit_resample(train_x, train_y)

        self.model.fit(train_x, train_y)
        self.test_X.append(np.vstack(test_x_ls))
        y_pred_ls, y_pred_ls_proba = self.predict_most_likely_class(test_x_ls)

        # 指标估计
        metrics = ModelEvaluatorSimple._metrics_calculation(test_y_ls, y_pred_ls)
        if self.roc is True:
            total_fpr_micro, total_tpr_micro, total_roc_auc_micro = (ModelEvaluator.
                                                                     _calculate_roc_curve(y_pred_ls_proba, test_y_ls))
            self.roc_result = (total_fpr_micro, total_tpr_micro, total_roc_auc_micro)
            total_fpr, total_tpr, total_roc_auc = (ModelEvaluator._calculate_roc_curve_class
                                                   (test_y_ls, y_pred_ls_proba))
            self.roc_class_result = (total_fpr, total_tpr, total_roc_auc)


        print("Accuracy:", metrics['accuracy'])
        print("Precision:", metrics['precision'])
        print("Recall:", metrics['recall'])
        print("F1 Score:", metrics['f1'])
        print("Specificity:", metrics['specificity'])
        print("Report:\n", metrics['report'])
        return metrics

    def predict_most_likely_class(self, test_X_ls):
        y_pred_ls = []
        y_pred_ls_proba = []
        n_classes = len(set(self.data_loader.train_data[1]))
        for test_X in test_X_ls:
            if self.classifier in ['logistic_l1', 'logistic_l2', 'knn', 'bayes', 'rf', 'mlp_2', 'mlp_4', 'mlp_8']:
                y_pred_prob = self.model.predict_proba(test_X)
                y_pred = np.argmax(y_pred_prob, axis=1)
            elif self.classifier in ['svm_l1', 'svm_l2']:
                y_pred = self.model.predict(test_X)
            elif self.classifier == 'lgbm':
                y_pred_prob = self.model.predict(test_X, num_iteration=self.model.best_iteration)
                y_pred = np.argmax(y_pred_prob, axis=1)
            elif self.classifier == 'xgb':
                y_pred_prob = self.model.predict(test_X, iteration_range=(0, self.model.best_iteration))
                y_pred = np.argmax(y_pred_prob, axis=1)
            else:
                raise ValueError("Unsupported classifier type for prediction.")
            counts = np.bincount(y_pred)
            # 如果 counts 的长度不足 n_classes，补齐为 0
            if len(counts) < n_classes:
                counts = np.pad(counts, (0, n_classes - len(counts)), 'constant')
            # 计算总计数
            total = np.sum(counts)
            # 计算每个类别的概率
            probabilities = counts / total
            y_pred_ls.append(np.argmax(counts))
            y_pred_ls_proba.append(probabilities)
        return y_pred_ls, y_pred_ls_proba

    @staticmethod
    def _calculate_roc_curve_class(y_pred_proba, y_true):

        n_classes = len(y_pred_proba[0])
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        y_true_bin = np.array(y_true_bin)
        y_pred_proba = np.array(y_pred_proba)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        return fpr, tpr, roc_auc

    @staticmethod
    def _calculate_roc_curve(y_pred_proba, y_true):
        print("calculate_roc_curve")
        n_classes = len(y_pred_proba[0])
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        y_true_bin = np.array(y_true_bin)
        y_pred_proba = np.array(y_pred_proba)

        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)

        return fpr_micro, tpr_micro, roc_auc_micro

    @staticmethod
    def _calculate_specificity(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn = cm[0, 0]
        fp = cm[0, 1]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return specificity

    @staticmethod
    def _check_confusion_matrix_size(cm):
        num_classes = 4

        adjusted_total_confusion_matrix = []
        if cm.shape != (num_classes, num_classes):

            new_cm = np.zeros((num_classes, num_classes))

            min_dim = min(cm.shape[0], num_classes)
            new_cm[:min_dim, :min_dim] = cm[:min_dim, :min_dim]
            adjusted_total_confusion_matrix = new_cm
        else:
            adjusted_total_confusion_matrix = cm
        return adjusted_total_confusion_matrix

    @staticmethod
    def _metrics_calculation(test_y_ls, y_pred_ls):
        print("Evaluating...")

        acc = accuracy_score(test_y_ls, y_pred_ls)
        precision = precision_score(test_y_ls, y_pred_ls, zero_division=0, average='macro')
        recall = recall_score(test_y_ls, y_pred_ls, zero_division=0, average='macro')
        f1 = f1_score(test_y_ls, y_pred_ls, zero_division=0, average='macro')
        report = classification_report(test_y_ls, y_pred_ls, zero_division=0)
        total_confusion_matrix = confusion_matrix(test_y_ls, y_pred_ls, normalize='true')
        specificity = ModelEvaluatorSimple._calculate_specificity(test_y_ls, y_pred_ls)
        # Calculate the mean normalized confusion matrix across all folds
        total_confusion_matrix = ModelEvaluatorSimple._check_confusion_matrix_size(total_confusion_matrix)

        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'report': report,
            'confusion_matrix': total_confusion_matrix
        }


if __name__ == '__main__':
    _back_to_root = "../.."

    activity_id = [1]
    data_path = "output/feature_selection"
    data_name = f"activity_{activity_id[0]}.csv"
    fold_groups_path = "input/feature_extraction"
    fold_groups_name = "fold_groups_new_with_combinations.csv"
    severity_mapping = {0: 0, 1: 1, 2: 2, 3: 3}
    single_data = PDDataLoader(activity_id, os.path.join(_back_to_root, data_path, data_name),
                               os.path.join(_back_to_root, fold_groups_path, fold_groups_name),
                               severity_mapping=severity_mapping)

    config = load_config(activity_id[0])
    classifier = 'mlp_2'
    # loading hyperparameters
    params = config[classifier]['params']
    print(params)
    model_trainer = ModelTrainer(classifier, params)
    study = ModelEvaluator(single_data, model_trainer)
    study.train_evaluate()
    study.shap_importance(_back_to_root)

    # sample as activity 14 15 16
    # comb_activity_id = [14, 15, 16]
    # classifier = 'lgbm'
    # comb_data_path = "output/activity_combination"
    # comb_data_name = "merged_activities_14_15_16_horizontal.csv"
    # fold_groups_path = "input/feature_extraction"
    # fold_groups_name = "fold_groups_new_with_combinations.csv"
    # severity_mapping = {0: 0, 1: 1, 2: 2, 3: 3}
    #
    # config = load_config(comb_activity_id[0])
    # # loading hyperparameters
    # params = config[classifier]['params']
    # print(params)
    #
    # comb_data = PDDataLoader(comb_activity_id, os.path.join(_back_to_root, comb_data_path, comb_data_name),
    #                          os.path.join(_back_to_root, fold_groups_path, fold_groups_name),
    #                          severity_mapping=severity_mapping)
    # model_trainer = ModelTrainer(classifier, params)
    # study = ModelEvaluator(comb_data, model_trainer)
    # study.train_evaluate()
    # study.shap_importance(_back_to_root)
