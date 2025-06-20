import os.path
import pandas as pd


def select_data(column_names, back_to_root, data_dir_path, data_file_name, feature_name_dir_path,
                feature_name_file_name, output_path):

    data_csv_path = os.path.join(data_dir_path, data_file_name)
    feature_name_csv_path = os.path.join(feature_name_dir_path, feature_name_file_name)

    try:

        df_read = pd.read_csv(feature_name_csv_path)
        data_read = pd.read_csv(data_csv_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except pd.errors.EmptyDataError as e:
        print(f"Error: {e}")
        return
    except pd.errors.ParserError as e:
        print(f"Error: {e}")
        return

    column_name_ls = []
    try:
        for column_name in column_names:
            column_name_ls.extend(df_read[column_name].tolist())
    except KeyError as e:
        print(f"Error: {e}")
        return

    column_name_ls.extend(['PatientID', 'activity_label', 'Severity_Level'])

    try:
        selected_data = data_read[column_name_ls]
    except KeyError as e:
        print(f"Error: {e}")
        return

    output_dir_path = os.path.join(back_to_root, output_path)
    output_name = '_'.join(column_names) + '_data.csv'
    output_name_path = os.path.join(output_dir_path, output_name)

    try:
        selected_data.to_csv(output_name_path, index=False)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    back_to_root = "../../.."
    data_dir_path = 'output/feature_extraction'
    data_file_name = 'wristr_acc_gyro_mag_feature_label.csv'
    feature_name_dir_path = 'input/feature_extraction/raw/'
    feature_name_file_name = 'feature_name.csv'
    output_path = 'output/select_sensors'

    select_data(
        column_names=['acc'],
        back_to_root=back_to_root,
        data_dir_path=os.path.join(back_to_root, data_dir_path),
        data_file_name=data_file_name,
        feature_name_dir_path=os.path.join(back_to_root, feature_name_dir_path),
        feature_name_file_name=feature_name_file_name,
        output_path=output_path
    )
