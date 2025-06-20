from src.module.severity_aeeseement import save_assessment_result
# First, make sure you have completed the feature extraction part
# training 12 model on 16 activity
_back_to_root = ".."
_data_path = "output/feature_selection"
fold_groups_path = "input/feature_extraction"
fold_groups_name = "fold_groups_new_with_combinations.csv"

classifiers = ['rf', 'xgb', 'lgbm', 'logistic_l1', 'logistic_l2', 'svm_l1', 'svm_l2', 'knn', 'bayes', 'mlp_2',
               'mlp_4', 'mlp_8']
activity_ids = list(range(1, 17))
save_assessment_result(_back_to_root, _data_path, activity_ids, classifiers, fold_groups_path, fold_groups_name)

# when you want to try the all activity on single model
# classifiers = ['xgb']
# activity_ids = list(range(1, 17))
# save_assessment_result(_back_to_root, activity_ids, classifiers)
