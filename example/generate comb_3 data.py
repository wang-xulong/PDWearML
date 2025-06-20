from src.module.severity_aeeseement.activity_combination_loader import ActivityCombLoader
import os

# Let's run the comb 3 (3 activity combination) performance
_back_to_root = ".."
base_path = "output/feature_selection"
activity_ids = [[i, 10, 11] for i in range(1, 17) if i not in [10, 11]]
activity_ids = [sorted(pair) for pair in activity_ids]
print("Loading data: ", activity_ids)

# generate the comb activities data,
# for instance, choose the comb way is 'horizontal'
# save data at base_path = "output/feature_selection"

for activity_id in activity_ids:
    loader = ActivityCombLoader(_back_to_root, base_path, activity_id, combination_mode='horizontal')
    loader.combine_and_save(os.path.join("output/activity_combination"))

# change the combination_mode as 'vertical' and 'weighted',
# Note that you should provide the weights as the format of list when you use the 'weighted' combination_mode,
#
# weighted = [
#     [0.544, 0.4874, 0.6808],
#     [0.5932, 0.4874, 0.6808],
#     [0.6505, 0.4874, 0.6808],
#     [0.562, 0.4874, 0.6808],
#     [0.5955, 0.4874, 0.6808],
#     [0.5145, 0.4874, 0.6808],
#     [0.5675, 0.4874, 0.6808],
#     [0.4736, 0.4874, 0.6808],
#     [0.631, 0.4874, 0.6808],
#     [0.4874, 0.6808, 0.577],
#     [0.4874, 0.6808, 0.577],
#     [0.4874, 0.6808, 0.4896],
#     [0.4874, 0.6808, 0.6158],
#     [0.4874, 0.6808, 0.5379]
# ]
# for activity_id, weight in zip(activity_ids, weighted):
#     loader = ActivityCombLoader(_back_to_root, base_path, activity_id, combination_mode='weighted')
#     loader.combine_and_save(os.path.join("output/activity_combination"), weighted=weight)





