from src.module.severity_aeeseement import SeverityAssessment, show_activity_shap_importance


_back_to_root = ".."

# example: activity_id = 11 (DRINK)
activity_id = 11
# First, set a severity assessment instance
_data_path = "output/feature_selection"
_data_name = f"activity_{activity_id}.csv"
sa = SeverityAssessment(_back_to_root, _data_path, _data_name, [activity_id], 'lgbm')

# Then, show its shap importance
show_activity_shap_importance(sa)



