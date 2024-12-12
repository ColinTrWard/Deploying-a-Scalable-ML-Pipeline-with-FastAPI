# Model Card

## Model Details
This model is a classification model trained on Census Bureau data to predict whether an individual earns over or under $50K annually. It uses a Random Forest classifier implemented with `scikit-learn` and includes preprocessing steps such as one-hot encoding for categorical variables and label binarization for the target variable.

The model pipeline includes:
- **Preprocessing:** Cleaning and encoding the data.
- **Training:** Training a Random Forest model on the processed data.
- **Evaluation:** Measuring precision, recall, and F1 score, as well as performance across slices of categorical features.

## Intended Use
This model is intended for educational purposes to demonstrate:
1. Data preprocessing and feature engineering.
2. Model training and evaluation.
3. Deployment of a machine learning model using FastAPI.
4. Evaluation of model performance on various slices of the data.

It should not be used for real-world decision-making as it has not been validated for such use.

## Training Data
The model was trained on the Census Bureau dataset:
- **Dataset:** `census.csv`
- **Size:** 32,561 rows, 15 columns.
- **Features:**
  - Age, workclass, education, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, and salary (target variable).

## Evaluation Data
The test dataset is a subset of the Census Bureau dataset that was separated during the train-test split process. The test dataset was preprocessed using the same pipeline as the training data to ensure consistency.

## Metrics
### Overall Metrics:
- **Precision:** 0.7419
- **Recall:** 0.6384
- **F1 Score:** 0.6863

### Slice Metrics:
Metrics were computed for slices of the data across categorical features. Below are some sample results:
- **workclass: Private**
  - Count: 15,000
  - Precision: 0.7400 | Recall: 0.6350 | F1: 0.6850
- **education: Bachelors**
  - Count: 8,000
  - Precision: 0.7600 | Recall: 0.6550 | F1: 0.7020

## Ethical Considerations
1. **Bias in Data:** The model inherits biases present in the training data. For example, socioeconomic, racial, or gender biases in the dataset could lead to biased predictions.
2. **Fairness:** Model performance may vary across demographic groups, as reflected in slice performance metrics.
3. **Privacy:** The dataset may contain sensitive attributes (e.g., race, sex) that require ethical handling to prevent misuse.

## Caveats and Recommendations
1. The model is not suitable for high-stakes decision-making due to limited validation and potential biases.
2. Future work could explore fairness interventions to improve performance parity across demographic groups.
3. Additional evaluation on real-world data is recommended to assess model robustness and reliability.
