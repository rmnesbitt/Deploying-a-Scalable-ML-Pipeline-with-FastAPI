import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.model import compute_model_metrics, train_model


def test_train_model():
    """Verify that train_model returns a Random Forest Classifier."""
    X_train = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [5, 6, 7, 8],
        }
    )
    y_train = [0, 1, 0, 1]

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics():
    """Verify that metric values are between 0 and 1."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_training_data_shape():
    """Verify that the training data has the expected shape."""
    X_train = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [5, 6, 7, 8],
        }
    )

    assert X_train.shape == (4, 2)
