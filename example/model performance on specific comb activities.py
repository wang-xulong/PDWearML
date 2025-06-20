from src.module.severity_aeeseement import save_comb_activity_assessment_result
from src.module.severity_aeeseement.activity_combination_loader import ActivityCombLoader
import os

_back_to_root = ".."
base_path = "output/feature_selection"
classifiers = ['rf', 'xgb', 'lgbm', 'logistic_l1', 'logistic_l2', 'svm_l1', 'svm_l2', 'knn', 'bayes', 'mlp_2',
               'mlp_4', 'mlp_8']

activity_ids = [[10, 11], [9, 10, 11], [7, 9, 10, 11]]
activity_ids = [sorted(pair) for pair in activity_ids]
print("Loading data: ", activity_ids)
# generate the comb activities data,
# for instance, choose the comb way is :
# horizontal
# vertical
# weighted
save_comb_activity_assessment_result(_back_to_root, activity_ids, classifiers, 'horizontal')


