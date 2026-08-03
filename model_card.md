# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This project uses a Random Forest Classifier to predict whether an individual's annual income is greater than $50,000 based on demographic and employment information from the U.S. Census Adult Income dataset. Before training, categorical features are one-hot encoded as part of the preprocessing pipeline.

## Intended Use
The model was created as part of the Udacity "Deploying a Scalable ML Pipeline with FastAPI" project. Its purpose is to demonstrate a complete machine learning workflow, including data preprocessing, model training, evaluation, testing, and deployment through a REST API.

It is intended for educational purposes only and should not be used to make real-world decisions involving employment, loans, insurance, or other high-impact situations.

## Training Data
The model was trained using the U.S. Census Adult Income dataset. The dataset includes information such as age, education, occupation, work class, marital status, race, sex, native country, hours worked per week, and capital gains or losses. The target variable is whether a person's annual income is above or below $50,000.

The data was split into an 80% training set and a 20% testing set before preprocessing and model training.

## Evaluation Data
The model was evaluated using the 20% test set created during the train-test split. The same preprocessing steps used for the training data were applied to the test data using the fitted encoder.

## Metrics
The model was evaluated using precision, recall, and F1 score. On the test dataset, the model achieved a precision of **0.7327**, a recall of **0.6397**, and an F1 score of **0.6830**.

In addition to the overall evaluation, the model's performance was measured across every unique value of each categorical feature. This slice analysis showed that performance varies across different demographic groups, with the greatest variation occurring for categories containing relatively few observations.

## Ethical Considerations
Because the dataset contains demographic information such as race, sex, marital status, and native country, the model may learn patterns that reflect historical biases in the data. For that reason, predictions from this model should not be used to make important decisions about individuals without additional fairness testing and careful review.

## Caveats and Recommendations
This model was built as part of a learning project and has several limitations. It has not undergone extensive hyperparameter tuning, and performance varies across some demographic groups, particularly those with relatively few examples.

Before using a model like this in a real application, it should be evaluated on additional data, tested for fairness and bias, and monitored regularly to ensure its predictions remain accurate over time.