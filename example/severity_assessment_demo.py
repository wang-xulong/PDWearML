import os
import sys

# get project path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
project_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
# put src into sys.path
sys.path.append(project_root)
sys.path.append(project_src)
from src.module.severity_aeeseement import save_assessment_result



classifiers = ['rf', 'xgb', 'lgbm', 'logistic_l1', 'logistic_l2', 'svm_l1', 'svm_l2', 'knn', 'bayes', 'mlp_2',
               'mlp_4', 'mlp_8']
activity_ids = list(range(1, 17))
save_assessment_result(project_root, activity_ids, classifiers)

