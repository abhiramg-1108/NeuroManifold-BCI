import warnings
import numpy as np
import moabb
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

warnings.filterwarnings("ignore")
moabb.set_log_level("error")


def load_subject_raw_epochs(subject_id: int = 1, fmin: float = 4.0, fmax: float = 38.0):
    """
    Loads real 4-class motor imagery EEG from BCI Competition IV - Dataset 2a.
    Returns train (Session 1) and test (Session 2) arrays of shape:
    (n_trials, 22 channels, n_timepoints).
    """
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4, fmin=fmin, fmax=fmax, resample=250)

    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[subject_id])

    train_mask = meta["session"] == "0train"
    test_mask = meta["session"] == "1test"

    X_train, y_train = X[train_mask], labels[train_mask]
    X_test, y_test = X[test_mask], labels[test_mask]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    print("Downloading/loading Subject 1 from BCI Competition IV-2a (BNCI2014_001)...")
    X_train, y_train, X_test, y_test = load_subject_raw_epochs(subject_id=1)

    print(f"Train shape (Session 1): {X_train.shape} | Classes: {np.unique(y_train)}")
    print(f"Test  shape (Session 2): {X_test.shape} | Classes: {np.unique(y_test)}")
    print("Data loaded cleanly — ready to build the Riemannian math engine!")