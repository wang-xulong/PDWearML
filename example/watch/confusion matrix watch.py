from src.module.severity_aeeseement import SeverityAssessment, get_confusion_matrices

import sys
import matplotlib.pyplot as plt
import os
import seaborn as sns
# get project path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
project_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'src'))
# put src into sys.path
sys.path.append(project_root)
sys.path.append(project_src)

_back_to_root = project_root

# example: activity_id = 1 (FT)
activity_id = 1
# First, set a severity assessment instance
_data_path = "output/feature_selection/watch"
fold_groups_path = "input/feature_extraction"
fold_groups_name = "watch_fold_groups_new_with_combinations.csv"
_data_name = f"activity_{activity_id}.csv"
sa = SeverityAssessment(_back_to_root, _data_path, _data_name, fold_groups_path, fold_groups_name,
                        [activity_id], 'lgbm', roc=True, watch=True)
confusion_matrix = get_confusion_matrices(sa)

# Class labels
class_labels = ['Normal', 'Mild', 'Moderate', 'Severe']

# Create confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', xticklabels=class_labels, yticklabels=class_labels,
            cbar=True, square=True, annot_kws={"size": 20})

# Set titles and labels
plt.xlabel('Predicted Label', fontsize=24)
plt.ylabel('True Label', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


output_shap_figure = os.path.join(_back_to_root,
                                              f'example/figure/confusion metrics on activity {activity_id} watch.png')

plt.savefig(output_shap_figure, bbox_inches='tight', dpi=300)
plt.tight_layout()
plt.show()

