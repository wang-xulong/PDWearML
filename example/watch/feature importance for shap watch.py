from src.module.severity_aeeseement import SeverityAssessment, show_activity_shap_importance
import os
import sys

# get project path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
project_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'src'))
# put src into sys.path
sys.path.append(project_root)
sys.path.append(project_src)

_back_to_root = project_root

# example: activity_id = 11 (DRINK)
activity_id = 1
# First, set a severity assessment instance
_data_path = "output/feature_selection/watch"
_data_name = f"activity_{activity_id}.csv"
fold_groups_path = "input/feature_extraction"
fold_groups_name = "watch_fold_groups_new_with_combinations.csv"
sa = SeverityAssessment(_back_to_root, _data_path, _data_name, fold_groups_path, fold_groups_name,
                        [activity_id], 'lgbm', watch=True)

# Then, show its shap importance
show_activity_shap_importance(sa)



