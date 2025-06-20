from src.module.severity_aeeseement import SeverityAssessment, get_roc_curve, get_roc_curve_class, get_confusion_matrices
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

_back_to_root = ".."

# example: activity_id = 11 (DRINK)
activity_id = 3
# First, set a severity assessment instance
_data_path = "output/feature_selection"
_data_name = f"activity_{activity_id}.csv"
sa_mlp = SeverityAssessment(_back_to_root, _data_path, _data_name, [activity_id], 'mlp_2', roc=True)
total_fpr_micro_mlp, total_tpr_micro_mlp, total_roc_auc_micro_mlp = get_roc_curve_class(sa_mlp)
confusion_matrix = get_confusion_matrices(sa_mlp)

# Class labels
class_labels = ['Normal', 'Mild', 'Moderate', 'Severe']

# Create confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, cbar=True, square=True)

# Set titles and labels
plt.xlabel('Predicted Label', fontsize=24)
plt.ylabel('True Label', fontsize=24)
plt.xticks(fontsize=20)  # 横轴刻度字体大小
plt.yticks(fontsize=20)  # 纵轴刻度字体大小


output_shap_figure = os.path.join(_back_to_root,
                                              f'example/figure/confusion metrics on activity {activity_id}.png')
# 显示图像
plt.savefig(output_shap_figure, bbox_inches='tight', dpi=300)
plt.tight_layout()
plt.show()
