# Customer Churn Prediction Model

This repository contains a machine learning model to predict customer churn for a telecommunications company. The model uses historical customer data to identify customers who are likely to churn and enables proactive retention strategies.

## Project Structure

- `customer_churn_data1.csv`: Sample dataset used for training the model.
- `churn.ipynb`: Jupyter notebook containing the full implementation of data preprocessing, model training, evaluation, and interpretation.
- `chunk_classifier.pkl`: Saved Random Forest model for real-time predictions.
- `README.md`: This readme file.
- `requirements.txt`: List of required Python packages.

## Setup Instructions

### Prerequisites

Ensure you have Python 3.8 or later installed on your system.

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/customer-churn-prediction.git
    cd customer-churn-prediction
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the `customer_churn_data1.csv` dataset and place it in the project directory.

### Usage

1. **Train and Save the Model**:

    - Open the `churn_model.ipynb` notebook.
    - Follow the steps to load the data, preprocess, train, and evaluate the model.
    - The trained Random Forest model is saved as `random_forest_model.pkl`.

2. **Load and Use the Saved Model**:

    ```python
    import joblib
    import pandas as pd

    # Load the saved model pipeline
    model_pipeline = joblib.load('random_forest_model.pkl')

    # Prepare new data for prediction (ensure it has the same structure as the training data)
    new_data = pd.DataFrame({
        'age': [28],
        'gender': ['Male'],
        'location': ['Urban'],
        'plan_type': ['Basic'],
        'tenure': [12],
        'monthly_charges': [75],
        'call_frequency': [100],
        'data_usage': [50],
        'internet_usage': [150],
        'payment_methods': ['Credit Card'],
        'issue_types': ['Billing'],
        'CLTV': [900],  # tenure * monthly_charges
        'ARPU': [75],   # monthly_charges
        'call_data_ratio': [100 / (100 + 50)]  # call_frequency / (call_frequency + data_usage)
    })

    # Make a prediction
    prediction = model_pipeline.predict(new_data)
    print("Churn Prediction:", prediction)
    ```

3. **Interpret Model Predictions**:

    - Use SHAP to interpret the model's predictions and understand the key factors influencing churn.

    ```python
    import shap

    # Load the saved model pipeline
    model_pipeline = joblib.load('random_forest_model.pkl')

    # Transform the test set using the preprocessor
    X_test_transformed = model_pipeline['preprocessor'].transform(new_data)

    # Generate SHAP values
    explainer = shap.Explainer(model_pipeline['model'])
    shap_values = explainer(X_test_transformed)

    # Visualize SHAP values
    shap.summary_plot(shap_values, features=X_test_transformed, feature_names=new_data.columns)
    ```

## Model Evaluation

### Performance Metrics

The performance of the model is evaluated using the following metrics:

- **Accuracy**: The proportion of correctly predicted instances out of the total instances.
- **Precision**: The proportion of positive identifications that were actually correct.
- **Recall**: The proportion of actual positives that were correctly identified.
- **F1-score**: The harmonic mean of precision and recall.
- **AUC-ROC**: The area under the Receiver Operating Characteristic curve.

### Model Comparison

Three machine learning algorithms were compared:

1. **Logistic Regression**
2. **Random Forest**
3. **XGBoost**

The Random Forest model performed the best in terms of accuracy and AUC-ROC, making it the chosen model for deployment.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

