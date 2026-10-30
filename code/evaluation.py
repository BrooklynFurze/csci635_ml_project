# evaluation shared by every text representation we compare (bag of words, tf-idf, and their lemmatized versions)
#
# review_score is 1 (positive) / -1 (negative) and ~82% of reviews are positive, so accuracy
# alone is misleading: always guessing "positive" already gets ~82% accuracy.
# macro F1 is the main metric: it averages the F1 of the positive and negative class equally,
# so a model only scores well if it actually catches the negative reviews too.
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix)

NEGATIVE, POSITIVE = -1, 1


def compute_metrics(y_true, y_pred, y_score):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        # average of the recall on each class (0.5 = no better than always guessing one class)
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        # how well the minority class (negative reviews) is handled
        "neg_precision": precision_score(y_true, y_pred, pos_label=NEGATIVE, zero_division=0),
        "neg_recall": recall_score(y_true, y_pred, pos_label=NEGATIVE),
        # threshold-free: how well the model ranks positive reviews above negative ones (0.5 = random)
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def predict(model, X):
    # y_score = predicted probability that the review is positive (needed for ROC AUC)
    y_score = model.predict_proba(X)[:, list(model.classes_).index(POSITIVE)]
    return model.predict(X), y_score


def cross_validate(vectorize, model, X, y, n_splits=5, seed=42):
    # vectorize(train_texts, val_texts) is re-fit on each training fold so the validation
    # fold's words / idf weights never leak into training.
    # same seed -> same folds for every method, so per-fold scores can be compared side by side
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, val_idx in folds.split(X, y):
        X_tr, X_val = vectorize(X.iloc[train_idx], X.iloc[val_idx])
        fold_model = clone(model).fit(X_tr, y.iloc[train_idx])
        scores.append(compute_metrics(y.iloc[val_idx], *predict(fold_model, X_val)))
    return pd.DataFrame(scores)


def test_evaluate(vectorize, model, X_train, X_test, y_train, y_test):
    X_tr, X_te = vectorize(X_train, X_test)
    final_model = clone(model).fit(X_tr, y_train)
    y_pred, y_score = predict(final_model, X_te)
    cm = confusion_matrix(y_test, y_pred, labels=[NEGATIVE, POSITIVE])
    return compute_metrics(y_test, y_pred, y_score), cm


def compare_methods(methods, y_train, y_test):
    # methods: list of (name, vectorize function, classifier, train texts, test texts) -- each method
    # brings its own texts so raw and lemmatized versions of the same split can sit in one table
    # CV on the training set compares the methods (mean ± std shows whether a gap is bigger than
    # fold-to-fold noise); the held-out test set is only used once per method, for the final numbers
    cv_summary, cv_f1, test_scores, matrices = {}, {}, {}, {}
    for name, vectorize, model, X_train, X_test in methods:
        print(f"evaluating {name}...")
        cv = cross_validate(vectorize, model, X_train, y_train)
        cv_summary[name] = cv.mean().map("{:.4f}".format) + " ± " + cv.std().map("{:.4f}".format)
        cv_f1[name] = cv["macro_f1"]
        test_scores[name], matrices[name] = test_evaluate(vectorize, model, X_train, X_test, y_train, y_test)

    test_table = pd.DataFrame(test_scores).T
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\n=== 5-fold CV on the training set (mean ± std) ===")
        print(pd.DataFrame(cv_summary).T)
        print("\n=== macro F1 per CV fold (same folds for every method) ===")
        print(pd.DataFrame(cv_f1).round(4))
        print("\n=== held-out test set ===")
        print(test_table.round(4))
    for name, cm in matrices.items():
        print(f"\nconfusion matrix on test set: {name}")
        print(pd.DataFrame(cm, index=["true neg", "true pos"], columns=["pred neg", "pred pos"]))
    return test_table
