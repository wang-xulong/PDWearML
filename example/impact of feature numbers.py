from src.module.feature_selection import FeatureSelection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline

back_to_root = ".."

# First, get acc_data.csv and put it into "output/select_sensors"


# Second, run FeatureSelection for acquiring the results after feature selection
# example: activity_id = 11 (DRINK)
activity_id = 11
fs = FeatureSelection(activity_id=activity_id, back_to_root=back_to_root, sensors=['acc'], seed=0)
# acquire the results after feature selection
fs.single_activity_feature_selection()
fs.save_important_feature()

# Third, acquire the impact of feature numbers
# This code will output the file "best_num_features_activity_11.csv"
fs.single_activity_best_num_features()

# Finally, plot it!
# get data
data = pd.read_csv(os.path.join(back_to_root, 'output/feature_selection',
                                f'best_num_features_activity_{activity_id}.csv'))


mean_scores = data[['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']].mean(axis=1)
std_scores = data[['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']].std(axis=1)

x_smooth = np.linspace(data['num_features'].min(), data['num_features'].max(), 300)
spl = make_interp_spline(data['num_features'], mean_scores, k=3)  # 使用三次样条插值
mean_scores_smooth = spl(x_smooth)
std_scores_smooth = make_interp_spline(data['num_features'], std_scores, k=3)(x_smooth)

plt.figure(figsize=(15, 6))
plt.plot(x_smooth, mean_scores_smooth, label='Mean F1 Score', color='b')
plt.fill_between(x_smooth, mean_scores_smooth - std_scores_smooth, mean_scores_smooth + std_scores_smooth, color='b', alpha=0.2, label='Standard Deviation')
plt.xlabel('Number of Features')
plt.ylabel('F1 Score')
plt.title('impact of feature number')
plt.legend()
plt.grid(True)

output_path = f'figure/impact of feature number on activity {activity_id}.png'
plt.savefig(output_path)
plt.show()

