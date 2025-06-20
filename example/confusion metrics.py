from src.module.severity_aeeseement import SeverityAssessment, get_roc_curve, get_roc_curve_class, get_confusion_matrices
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

from proplot import rc
rc["font.family"] = "sans-serif"
rc['font.sans-serif'] = 'TeX Gyre Heros'
rc["axes.labelsize"] = 24
rc["tick.labelsize"] = 20
rc["suptitle.size"] = 15
rc["title.size"] = 26
rc["tick.minor"] = False

_back_to_root = ".."
fold_groups_path = "input/feature_extraction"
fold_groups_name = "fold_groups_new_with_combinations.csv"
# example: activity_id = 11 (DRINK)
activity_id = 11
# First, set a severity assessment instance
_data_path = "output/feature_selection"
_data_name = f"activity_{activity_id}.csv"
sa_mlp = SeverityAssessment(_back_to_root, _data_path, _data_name, fold_groups_path,fold_groups_name,
[activity_id], 'mlp_2', roc=True)
confusion_matrix = get_confusion_matrices(sa_mlp)

# activity_id = [9, 10, 11]
# # First, set a severity assessment instance
# combination_mode = 'horizontal'
# _data_path = "output/activity_combination"
#
# activity_ids_str = "_".join(map(str, activity_id))
# _file_name = f"merged_activities_{activity_ids_str}_{combination_mode}.csv"
# sa_mlp = SeverityAssessment(_back_to_root, _data_path, _file_name, fold_groups_path,fold_groups_name, activity_id,
#                             'mlp_2', roc=True)
# total_fpr_micro_mlp, total_tpr_micro_mlp, total_roc_auc_micro_mlp = get_roc_curve_class(sa_mlp)
# confusion_matrix = get_confusion_matrices(sa_mlp)

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
plt.title('DRINK')


output_shap_figure = os.path.join(_back_to_root,
                                              f'example/figure/confusion metrics on activity {activity_id}.png')

plt.savefig(output_shap_figure, bbox_inches='tight', dpi=300)
plt.tight_layout()
plt.show()
