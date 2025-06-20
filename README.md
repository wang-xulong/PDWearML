# Wearable-based Machine Learning using Combined Daily Activities Combination for Fast-Assessment of Free-living Parkinson’s Disease


This project provides a comprehensive workflow for estimating the severity of Parkinson's Disease (PD) using signals from wearable sensors. The project is modular, with each module focusing on different aspects of the data processing and analysis pipeline, from feature extraction to severity assessment.

## Project Structure

The project is organized into the following modules, each located in the `src/module/` directory:

- **feature_extraction/**
  - **`FeatureExtraction.py`**: This script extracts relevant features from the raw sensor signals. These features are crucial for downstream tasks such as classification and severity assessment.
  - **`utils/`**: Contains utility scripts (`tremor_utils.py`, `pd_utils.py`) that support the feature extraction process, providing functions for signal processing and feature computation.
- **select_sensors/**
  - **`select_sensors.py`**: This script is responsible for selecting the most relevant sensors for the analysis. Sensor selection is critical for optimizing the performance of the severity estimation model while reducing computational complexity.
- **feature_selection/**
  - **`FeatureSelection.py`**: Provides methods for selecting the most significant features from the dataset. Effective feature selection can improve the model's accuracy and efficiency.
- **severity_aeeseement/**
  - **`severity_assessment.py`**: Implements the core logic for assessing the severity of PD based on the extracted features. The assessment could be based on predefined criteria or machine learning models.
  - **`activity_combination_loader.py`**: Loads and processes combinations of activities from sensor data to be used in the severity assessment.

## Setup and Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sparksoftltd/huawei-smartwatch-parkingson-disease-detection.git
   cd huawei-smartwatch-parkingson-disease-detection
   ```

2. **Create a conda environment and install dependencies**:
   
   ```bash
   conda create --name pd-severity-env python=3.8
   conda activate pd-severity-env
   pip install -r requirements.txt

3. **Download the data and organised it into `input/` and `output/` folder.**
   
   Ensure that the data of `input/feature_extraction` exists if you would like to execute the whole process below.




## Usage

Each module can be run independently depending on the task you need to perform. Below are some usage examples:

- **Feature Extraction**:

  ```bash
  python example/feature_extraction_demo.py
  ```
  This command will extract features from your sensor data.


- **Sensor Selection**:

  ```bash
  python example/select_sensors_demo.py
  ```

  Run this script to perform sensor selection.


- **Feature Selection**:

  ```bash
  python example/feature_selection_demo.py
  ```

  Run this script to perform feature selection.


- **Severity assessment**:

  ```bash
  python example/severity_aeeseement_demo.py
  ```

  Run this script to perform severity assessment (Take all activity on 12 algorithms for example).



## Contributing

We welcome contributions from the community! If you'd like to contribute, please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For questions or support, please contact [xl.wang@sheffield.ac.uk](mailto:your-email@example.com).
