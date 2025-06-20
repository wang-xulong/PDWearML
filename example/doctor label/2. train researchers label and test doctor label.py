import os
import sys
import pandas as pd
from src.utils.ModelTrainerEvaluator import ModelTrainer, ModelEvaluatorSimple
from src.utils.PDDataLoader import PDDataLoaderSimple
import yaml
import pandas as pd
from datetime import datetime

# get project path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# put src into sys.path
sys.path.append(project_root)


def load_config(activity_id: int):
    config_path = os.path.join(project_root, f'src/utils/config_doctor_label/activity_{activity_id}.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)




_dir_path = os.path.join(project_root, "output/feature_selection/doctor_label")
classifier = 'mlp_4'
save_data = "output/severity_assessment"
results_df = []
for a in range(1, 17):
    researcher_data = PDDataLoaderSimple([a], _dir_path, valid=True)
    doctor_data = PDDataLoaderSimple([a], _dir_path, valid=False)
    config = load_config(a)
    params = config[classifier]['params']
    print('researcher assessing')

    researcher_model_trainer = ModelTrainer(classifier, params)
    researcher_study = ModelEvaluatorSimple(researcher_data, researcher_model_trainer)
    researcher_metrics = researcher_study.train_evaluate()
    row1 = {
        'activity_id': a,
        'classifier': classifier,
        'rater': 'researcher',
        'acc': researcher_metrics['accuracy'],
        'precision': researcher_metrics['precision'],
        'recall': researcher_metrics['recall'],
        'f1': researcher_metrics['f1'],
        'specificity': researcher_metrics['specificity']
    }
    print('doctor assessing')
    doctor_model_trainer = ModelTrainer(classifier, params)
    doctor_study = ModelEvaluatorSimple(doctor_data, doctor_model_trainer)
    doctor_metrics = doctor_study.train_evaluate()
    row2 = {
        'activity_id': a,
        'classifier': classifier,
        'rater': 'doctor',
        'acc': doctor_metrics['accuracy'],
        'precision': doctor_metrics['precision'],
        'recall': doctor_metrics['recall'],
        'f1': doctor_metrics['f1'],
        'specificity': doctor_metrics['specificity']
    }

    results_df.append(row1)
    results_df.append(row2)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
results_df = pd.DataFrame(results_df)
results_df.to_csv(os.path.join(project_root, save_data,
                               f'doctor_researcher_metrics_{current_time}.csv'), index=False)
print(f"All results have been saved to 'doctor_researcher_metrics_{current_time}.csv'")



