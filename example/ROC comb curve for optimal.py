from src.module.severity_aeeseement import SeverityAssessment, get_roc_curve, get_roc_curve_class, get_confusion_matrices
import numpy as np
import matplotlib.pyplot as plt
import os

_back_to_root = ".."

activity_id = [9, 10, 11]
# First, set a severity assessment instance
combination_mode = 'horizontal'
_data_path = "output/activity_combination"
activity_ids_str = "_".join(map(str, activity_id))
_file_name = f"merged_activities_{activity_ids_str}_{combination_mode}.csv"
sa_mlp = SeverityAssessment(_back_to_root, _data_path, _file_name, activity_id, 'mlp_2', roc=True)
total_fpr_micro_mlp, total_tpr_micro_mlp, total_roc_auc_micro_mlp = get_roc_curve_class(sa_mlp)
print(get_confusion_matrices(sa_mlp))

# detail_model_name = {'MLP': 'mlp_8', 'Linear': 'logistic_l1', 'KNN': 'knn', 'Bayes': 'bayes', 'Tree': 'lgbm'}


class_mapping = {0: 'Normal', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}


plt.figure(figsize=(10, 8))


colors = ['blue', 'green', 'orange', 'red']
linestyles = ['-', '--', '-.', ':']


for class_idx in range(4):

    mean_fpr = np.linspace(0, 1, 100)
    tpr_resampled = []

    aucs = []

    for fold_num in range(5):
        fpr = total_fpr_micro_mlp[fold_num][class_idx]
        tpr = total_tpr_micro_mlp[fold_num][class_idx]
        if not np.isnan(fpr).any() and not np.isnan(tpr).any():
            tpr_resampled.append(np.interp(mean_fpr, fpr, tpr))


            auc_value = total_roc_auc_micro_mlp[fold_num][class_idx]
            aucs.append(auc_value)
        else:

            print(f"Skipping fold {fold_num} for class {class_mapping[class_idx]} due to NaN values")


    if tpr_resampled:
        mean_tpr = np.mean(tpr_resampled, axis=0)
        std_tpr = np.std(tpr_resampled, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        plt.plot(mean_fpr, mean_tpr, color=colors[class_idx], linestyle=linestyles[class_idx],
                 lw=2, label=f'{class_mapping[class_idx]} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')


        plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color=colors[class_idx], alpha=0.2)


# plt.plot([0, 1], [0, 1], 'k--', lw=2)


plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('ROC curves for [WALK,AFC,DRINK]', fontsize=24)
plt.legend(loc="lower right", fontsize=20)
plt.grid(True)


output_shap_figure = os.path.join(_back_to_root, f'example/figure/ROC class on activity {activity_id}.png')
plt.savefig(output_shap_figure, bbox_inches='tight', dpi=300)
plt.tight_layout()
plt.show()


