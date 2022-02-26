# Run the Hello World! version of the ML packages
import numpy as np


def test_sklearn():
    # first example from sklearn docs
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(random_state=0)
    X = [[1, 2, 3], [11, 12, 13]]  # 2 samples, 3 features
    y = [0, 1]  # classes of each sample
    clf.fit(X, y)
    prediction = clf.predict([[4, 5, 6], [14, 15, 16]])
    assert np.allclose(prediction, np.array([0, 1]))
