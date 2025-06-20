from src.module.severity_aeeseement import SeverityAssessment, get_confusion_matrices

_back_to_root = ".."

# example: activity_id = 11 (DRINK)
activity_id = 11
# First, set a severity assessment instance
_data_path = "output/feature_selection"
_data_name = f"activity_{activity_id}.csv"
sa = SeverityAssessment(_back_to_root, _data_path, _data_name, [activity_id], 'lgbm', roc=True)
a = get_confusion_matrices(sa)

print(a)
