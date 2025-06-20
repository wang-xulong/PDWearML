import yaml
import os
from src.utils.PDDataLoader import PDDataLoader
from src.utils.ModelTrainerEvaluator import ModelTrainer, ModelEvaluator
import pandas as pd
from datetime import datetime
from typing import List
from src.utils.utils import set_seed
import copy

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class SeverityAssessment:
    def __init__(self, back_to_root: str, data_path: str, data_name: str, fold_groups_path: str, fold_groups_name: str,
                 activity_id: List, classifier: str, **kwargs):
        self.back_to_root = back_to_root
        self.activity_id = activity_id
        self.classifier = classifier
        self.data_path = data_path
        self.data_name = data_name
        self.fold_groups_path = fold_groups_path
        self.fold_groups_name = fold_groups_name
        self.shap_importance = False
        self.data_loader = PDDataLoader(activity_id, os.path.join(self.back_to_root, self.data_path, self.data_name),
                                        os.path.join(self.back_to_root, self.fold_groups_path, self.fold_groups_name))
        self.single_activity = len(activity_id) == 1
        self.roc = kwargs.get('roc', False)
        self.watch = kwargs.get('watch', False)
        self.model_evaluator = None
        set_seed(0)

    def load_config(self):
        if self.single_activity:
            config_path = f'src/utils/config/activity_{self.activity_id[0]}.yaml'
        else:
            config_path = f'src/utils/config/comb_{len(self.activity_id)}.yaml'
        with open(os.path.join(self.back_to_root, config_path), 'r') as file:
            return yaml.safe_load(file)

    def load_config_watch(self):
        if self.single_activity:
            config_path = f'src/utils/config_watch/activity_{self.activity_id[0]}.yaml'
        else:
            config_path = f'src/utils/config_watch/comb_{len(self.activity_id)}.yaml'
        with open(os.path.join(self.back_to_root, config_path), 'r') as file:
            return yaml.safe_load(file)

    def assessment(self):
        # loading config
        if self.watch is False:
            config = self.load_config()
        else:
            config = self.load_config_watch()
        # print(type(config['mlp_2']['params']['alpha']))
        if self.single_activity:
            assert self.activity_id[0] == config['activity_id'], "error activity_id module"
        else:
            assert 'comb_' + str(len(self.activity_id)) == config['activity_id'], "error activity_id module"
        # loading hyperparameters
        params = config[self.classifier]['params']
        print(params)
        # severity assessment
        print(f"classifier [{self.classifier}] on activity_id {config['activity_id']}")
        model_trainer = ModelTrainer(self.classifier, params)
        study = ModelEvaluator(self.data_loader, model_trainer, roc=self.roc)
        metrics = study.train_evaluate()
        if self.shap_importance:
            study.shap_importance(self.back_to_root)

        self.model_evaluator = copy.deepcopy(study)
        return metrics


def get_roc_curve(severity_assessment: SeverityAssessment):
    severity_assessment.roc = True
    severity_assessment.assessment()
    return severity_assessment.model_evaluator.roc_result


def get_roc_curve_class(severity_assessment: SeverityAssessment):
    severity_assessment.roc = True
    severity_assessment.assessment()
    return severity_assessment.model_evaluator.roc_class_result


def get_confusion_matrices(severity_assessment: SeverityAssessment):
    metrics = severity_assessment.assessment()
    return metrics.get('mean_confusion_matrix', False)


def show_activity_shap_importance(severity_assessment: SeverityAssessment):
    severity_assessment.shap_importance = True
    severity_assessment.assessment()


def save_assessment_result(back_to_root: str, data_path: str, activity_list: list, classifier_list: list,
                           fold_groups_path: str, fold_groups_name: str, **kwargs):
    # data_path = "output/feature_selection"

    results = []

    for c in classifier_list:
        for a in activity_list:

            data_name = f"activity_{a}.csv"
            watch = kwargs.get('watch', False)
            sa = SeverityAssessment(back_to_root, data_path, data_name, fold_groups_path, fold_groups_name,
                                    [a], str(c), watch=watch)
            metrics = sa.assessment()


            row = {
                'activity_id': a,
                'classifier': c,
                'acc_mean': metrics['mean_accuracy'],
                'precision_mean': metrics['mean_precision'],
                'recall_mean': metrics['mean_recall'],
                'f1_mean': metrics['mean_f1'],
                'specificity_mean': metrics['mean_specificity'],
            }


            for fold_metric in metrics['fold_metrics']:
                fold_num = fold_metric['fold_num']
                row[f'acc_fold_{fold_num}'] = fold_metric['accuracy']
                row[f'precision_fold_{fold_num}'] = fold_metric['precision']
                row[f'recall_fold_{fold_num}'] = fold_metric['recall']
                row[f'f1_fold_{fold_num}'] = fold_metric['f1']
                row[f'specificity_fold_{fold_num}'] = fold_metric['specificity']


            results.append(row)

    save_data_path = "output/severity_assessment"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(back_to_root, save_data_path,
                                   f'activity_classifier_metrics_{current_time}.csv'), index=False)
    print(f"All results have been saved to 'activity_classifier_metrics_{current_time}.csv'")


def save_comb_activity_assessment_result(back_to_root: str, activity_list: list, classifier_list: list,
                                         combination_mode: str):
    data_path = "output/activity_combination"

    results = []

    for c in classifier_list:
        for a in activity_list:
            assert isinstance(c, str), "error classifier type"
            assert isinstance(a, List), "combination should be a list type"
            activity_ids_str = "_".join(map(str, a))
            file_name = f"merged_activities_{activity_ids_str}_{combination_mode}.csv"
            comb_sa = SeverityAssessment(back_to_root, data_path, file_name, a, str(c))
            comb_metrics = comb_sa.assessment()
            row = {
                'activity_id': a,
                'classifier': c,
                'acc_mean': comb_metrics['mean_accuracy'],
                'precision_mean': comb_metrics['mean_precision'],
                'recall_mean': comb_metrics['mean_recall'],
                'f1_mean': comb_metrics['mean_f1'],
                'specificity_mean': comb_metrics['mean_specificity'],
            }
            for fold_metric in comb_metrics['fold_metrics']:
                fold_num = fold_metric['fold_num']
                row[f'acc_fold_{fold_num}'] = fold_metric['accuracy']
                row[f'precision_fold_{fold_num}'] = fold_metric['precision']
                row[f'recall_fold_{fold_num}'] = fold_metric['recall']
                row[f'f1_fold_{fold_num}'] = fold_metric['f1']
                row[f'specificity_fold_{fold_num}'] = fold_metric['specificity']


            results.append(row)
    save_data_path = "output/severity_assessment"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(back_to_root, save_data_path,
                                   f'activity_combination_{combination_mode}_classifier_metrics_{current_time}.csv'),
                      index=False)
    print(
        f"All results have been saved to 'activity_combination_{combination_mode}classifier_metrics_{current_time}.csv'")


if __name__ == '__main__':
    _back_to_root = "../../.."

    activity_id = [1]
    _data_path = "output/feature_selection"
    _data_name = f"activity_{activity_id[0]}.csv"
    fold_groups_path = "input/feature_extraction"
    fold_groups_name = "fold_groups_new_with_combinations.csv"

    sa = SeverityAssessment(_back_to_root, _data_path, _data_name, fold_groups_path, fold_groups_name,
                            activity_id, 'xgb')
    sa.assessment()

__all__ = [
    'SeverityAssessment',
    'show_activity_shap_importance',
    'save_assessment_result',
    'save_comb_activity_assessment_result',
    'get_roc_curve',
    'get_roc_curve_class',
    'get_confusion_matrices'
]
