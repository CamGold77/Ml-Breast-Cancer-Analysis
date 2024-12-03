# Breast Cancer Prediction

This project is focused on using machine learning to predict whether a tumor is benign or malignant based on features extracted from breast cancer data. The dataset used for this analysis is the **Breast Cancer Wisconsin (Diagnostic) Data Set** from the UCI Machine Learning Repository.

## Project Overview

The goal of this project is to:
- Analyze breast cancer diagnostic data.
- Build a machine learning model that classifies tumors as malignant or benign.
- Use data preprocessing, feature selection, and classification algorithms to improve model accuracy.

This project demonstrates data preprocessing, exploratory data analysis (EDA), and machine learning classification using tools like:
- **Python**: The primary programming language for data analysis and machine learning.
- **pandas**: For data manipulation and exploration.
- **matplotlib** and **seaborn**: For data visualization.
- **scikit-learn**: For building and evaluating machine learning models.

## Dataset

The dataset used in this project is the **Breast Cancer Wisconsin (Diagnostic) Data Set**, which contains the following columns:
- `id`: The unique identifier for each tumor.
- `diagnosis`: The diagnosis of the tumor (M = malignant, B = benign).
- Various numerical features (e.g., `radius_mean`, `texture_mean`, `perimeter_mean`, etc.) that describe characteristics of the tumor cells.

## Requirements

To run this project, you need to have the following Python libraries installed:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install them using `pip`: 

## Usage

1. Clone this repository:
```bash
git clone https://github.com/CamGold77/Ml-Breast-Cancer-Analysis.git cd Ml-Breast-Cancer-Analysis
```
2. Open the `breast.ipynb` notebook in a Jupyter environment:

```bash
3. Open the `breast.ipynb` notebook in a Jupyter environment:

4. Follow the steps in the notebook to:
   - Load the dataset.
   - Perform exploratory data analysis (EDA).
   - Preprocess the data (e.g., handle missing values, scale features).
   - Train and evaluate a machine learning model for tumor classification.
```
## Project Flow

1. **Data Loading**:
   - The dataset is loaded into a pandas DataFrame for easy manipulation.
   - The data is analyzed to identify missing values and other preprocessing needs.

2. **Exploratory Data Analysis (EDA)**:
   - Visualizations (e.g., histograms, scatter plots) are used to explore relationships between features and the target variable (`diagnosis`).
   - Correlations between features are examined.

3. **Data Preprocessing**:
   - Categorical features are encoded (e.g., converting `M` and `B` labels to numerical values).
   - Feature scaling is applied to normalize numerical features.
   - The dataset is split into training and testing sets.

4. **Model Training**:
   - A machine learning model (e.g., logistic regression, random forest, or support vector machine) is trained using the training data.
   - The model is evaluated using accuracy, precision, recall, and F1 score metrics.

5. **Model Evaluation**:
   - Visualizations such as confusion matrices and ROC curves are used to evaluate model performance.
   - Suggestions for improving the model are discussed based on evaluation results.

## Results

The trained machine learning model is capable of classifying breast cancer tumors as either **benign** or **malignant** with a certain level of accuracy. The evaluation metrics (accuracy, precision, recall, F1 score) will provide insight into how well the model performs. You can adjust model parameters and explore different algorithms to improve performance further.

5. Open a pull request to merge changes into the main branch.

## License

This project is licensed under the MIT License. See the LICENSE file for details.



