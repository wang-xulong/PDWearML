from src.module.severity_aeeseement import SeverityAssessment, get_roc_curve, get_roc_curve_class
import numpy as np
import matplotlib.pyplot as plt
import os

_back_to_root = ".."

# example: activity_id = 11 (DRINK)
activity_id = [9, 10, 11]
# First, set a severity assessment instance
combination_mode = 'horizontal'
_data_path = "output/activity_combination"
activity_ids_str = "_".join(map(str, activity_id))
_file_name = f"merged_activities_{activity_ids_str}_{combination_mode}.csv"
sa_mlp = SeverityAssessment(_back_to_root, _data_path, _file_name, activity_id, 'mlp_2', roc=True)
total_fpr_micro_mlp, total_tpr_micro_mlp, total_roc_auc_micro_mlp = get_roc_curve(sa_mlp)

sa_lgbm = SeverityAssessment(_back_to_root, _data_path, _file_name, activity_id, 'lgbm', roc=True)
total_fpr_micro_lgbm, total_tpr_micro_lgbm, total_roc_auc_micro_lgbm = get_roc_curve(sa_lgbm)

sa_linear = SeverityAssessment(_back_to_root, _data_path, _file_name, activity_id, 'logistic_l1', roc=True)
total_fpr_micro_linear, total_tpr_linear, total_roc_auc_linear = get_roc_curve(sa_linear)

sa_bayes = SeverityAssessment(_back_to_root, _data_path, _file_name, activity_id, 'bayes', roc=True)
total_fpr_micro_bayes, total_tpr_micro_bayes, total_roc_auc_micro_bayes = get_roc_curve(sa_bayes)

sa_knn = SeverityAssessment(_back_to_root, _data_path, _file_name, activity_id, 'knn', roc=True)
total_fpr_micro_knn, total_tpr_micro_knn, total_roc_auc_micro_knn = get_roc_curve(sa_knn)

detail_model_name = {'MLP': 'mlp_2', 'Linear': 'logistic_l1', 'KNN': 'knn', 'Bayes': 'bayes', 'Tree': 'lgbm'}

models = {
    'MLP': {
        'total_fpr_micro': total_fpr_micro_mlp,
        'total_tpr_micro': total_tpr_micro_mlp,
        'total_roc_auc_micro': total_roc_auc_micro_mlp
    },
    'Linear': {
        'total_fpr_micro': total_fpr_micro_linear,
        'total_tpr_micro': total_tpr_linear,
        'total_roc_auc_micro': total_roc_auc_linear
    },
    'KNN': {
        'total_fpr_micro': total_fpr_micro_knn,
        'total_tpr_micro': total_tpr_micro_knn,
        'total_roc_auc_micro': total_roc_auc_micro_knn
    },
    'Bayes': {
        'total_fpr_micro': total_fpr_micro_bayes,
        'total_tpr_micro': total_tpr_micro_bayes,
        'total_roc_auc_micro': total_roc_auc_micro_bayes
    },
    'Tree': {
        'total_fpr_micro': total_fpr_micro_lgbm,
        'total_tpr_micro': total_tpr_micro_lgbm,
        'total_roc_auc_micro': total_roc_auc_micro_lgbm
    }
}

plt.figure(figsize=(10, 8))

for model_name, model_data in models.items():
    total_fpr_micro = model_data['total_fpr_micro']
    total_tpr_micro = model_data['total_tpr_micro']
    total_roc_auc_micro = model_data['total_roc_auc_micro']


    mean_fpr = np.linspace(0, 1, 100)

    # Interpolating tpr values to match the consistent fpr range
    tpr_resampled = []
    for fpr, tpr in zip(total_fpr_micro, total_tpr_micro):
        # Interpolate the TPR values for consistent FPR points
        tpr_resampled.append(np.interp(mean_fpr, fpr, tpr))

    mean_tpr = np.mean(tpr_resampled, axis=0)
    std_tpr = np.std(tpr_resampled, axis=0)


    mean_auc = np.mean(total_roc_auc_micro)
    std_auc = np.std(total_roc_auc_micro)


    plt.plot(mean_fpr, mean_tpr, lw=2, alpha=.8, label=f'{detail_model_name[model_name]} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')

    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, alpha=.2)


# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)


plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('ROC curves for [WALK,AFC,DRINK]', fontsize=24)
plt.legend(loc="lower right", fontsize=20)
plt.grid(alpha=0.3)
output_shap_figure = os.path.join(_back_to_root,
                                              f'example/figure/ROC on activity {activity_id}.png')

plt.savefig(output_shap_figure, bbox_inches='tight', dpi=300)
plt.tight_layout()
plt.show()
