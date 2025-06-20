from src.module.severity_aeeseement import save_comb_activity_assessment_result
from src.module.severity_aeeseement.activity_combination_loader import ActivityCombLoader
import os

_back_to_root = ".."
base_path = "output/feature_selection"
classifiers = ['mlp_2']


# First, Let's run the comb 2 (2 activity combination) performance
activity_ids = [[i, 11] for i in range(1, 17) if i != 11]
activity_ids = [sorted(pair) for pair in activity_ids]
print("Loading data: ", activity_ids)
# generate the comb activities data,
# for instance, choose the comb way is :
# horizontal
# vertical
# weighted
save_comb_activity_assessment_result(_back_to_root, activity_ids, classifiers, 'horizontal')


"""
# Second, Let's run the comb 3 (3 activity combination) performance
activity_ids = [[i, 10, 11] for i in range(1, 17) if i not in [10, 11]]
activity_ids = [sorted(pair) for pair in activity_ids]
print("Loading data: ", activity_ids)
# generate the comb activities data,
# for instance, choose the comb way is :
# horizontal
# vertical
# weighted
save_comb_activity_assessment_result(_back_to_root, activity_ids, classifiers, 'weighted')


# Third, Let's run the comb 4 (4 activity combination) performance
activity_ids = [[i, 9, 10, 11] for i in range(1, 17) if i not in [9, 10, 11]]
activity_ids = [sorted(pair) for pair in activity_ids]
print("Loading data: ", activity_ids)
# generate the comb activities data,
# for instance, choose the comb way is :
# horizontal
# vertical
# weighted
save_comb_activity_assessment_result(_back_to_root, activity_ids, classifiers, 'weighted')

"""