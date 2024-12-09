import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.utils import check_consistent_length, check_array, column_or_1d


def hinge_loss(y_true, pred_decision, *, labels=None, sample_weight=None):
    """Average hinge loss (non-regularized).

    In binary class case, assuming labels in y_true are encoded with +1 and -1,
    when a prediction mistake is made, ``margin = y_true * pred_decision`` is
    always negative (since the signs disagree), implying ``1 - margin`` is
    always greater than 1.  The cumulated hinge loss is therefore an upper
    bound of the number of mistakes made by the classifier.

    In multiclass case, the function expects that either all the labels are
    included in y_true or an optional labels argument is provided which
    contains all the labels. The multilabel margin is calculated according
    to Crammer-Singer's method. As in the binary case, the cumulated hinge loss
    is an upper bound of the number of mistakes made by the classifier.

    Read more in the :ref:`User Guide <hinge_loss>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target, consisting of integers of two values. The positive label
        must be greater than the negative label.

    pred_decision : array-like of shape (n_samples,) or (n_samples, n_classes)
        Predicted decisions, as output by decision_function (floats).

    labels : array-like, default=None
        Contains all the labels for the problem. Used in multiclass hinge loss.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float
        Average hinge loss.

    References
    ----------
    .. [1] `Wikipedia entry on the Hinge loss
           <https://en.wikipedia.org/wiki/Hinge_loss>`_.

    .. [2] Koby Crammer, Yoram Singer. On the Algorithmic
           Implementation of Multiclass Kernel-based Vector
           Machines. Journal of Machine Learning Research 2,
           (2001), 265-292.

    .. [3] `L1 AND L2 Regularization for Multiclass Hinge Loss Models
           by Robert C. Moore, John DeNero
           <https://storage.googleapis.com/pub-tools-public-publication-data/pdf/37362.pdf>`_.

    Examples
    --------
    >>> from sklearn import svm
    >>> from sklearn.metrics import hinge_loss
    >>> X = [[0], [1]]
    >>> y = [-1, 1]
    >>> est = svm.LinearSVC(dual="auto", random_state=0)
    >>> est.fit(X, y)
    LinearSVC(dual='auto', random_state=0)
    >>> pred_decision = est.decision_function([[-2], [3], [0.5]])
    >>> pred_decision
    array([-2.18...,  2.36...,  0.09...])
    >>> hinge_loss([-1, 1, 1], pred_decision)
    0.30...

    In the multiclass case:

    >>> import numpy as np
    >>> X = np.array([[0], [1], [2], [3]])
    >>> Y = np.array([0, 1, 2, 3])
    >>> labels = np.array([0, 1, 2, 3])
    >>> est = svm.LinearSVC(dual="auto")
    >>> est.fit(X, Y)
    LinearSVC(dual='auto')
    >>> pred_decision = est.decision_function([[-1], [2], [3]])
    >>> y_true = [0, 2, 3]
    >>> hinge_loss(y_true, pred_decision, labels=labels)
    0.56...
    """
    check_consistent_length(y_true, pred_decision, sample_weight)
    pred_decision = check_array(pred_decision, ensure_2d=False)
    y_true = column_or_1d(y_true)
    y_true_unique = np.unique(labels if labels is not None else y_true)

    if y_true_unique.size > 2:
        if pred_decision.ndim <= 1:
            raise ValueError(
                "The shape of pred_decision cannot be 1d array"
                "with a multiclass target. pred_decision shape "
                "must be (n_samples, n_classes), that is "
                f"({y_true.shape[0]}, {y_true_unique.size})."
                f" Got: {pred_decision.shape}"
            )

        # pred_decision.ndim > 1 is true
        if y_true_unique.size != pred_decision.shape[1]:
            if labels is None:
                raise ValueError(
                    "Please include all labels in y_true "
                    "or pass labels as third argument"
                )
            else:
                raise ValueError(
                    "The shape of pred_decision is not "
                    "consistent with the number of classes. "
                    "With a multiclass target, pred_decision "
                    "shape must be "
                    "(n_samples, n_classes), that is "
                    f"({y_true.shape[0]}, {y_true_unique.size}). "
                    f"Got: {pred_decision.shape}"
                )
        if labels is None:
            labels = y_true_unique
        le = LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        mask = np.ones_like(pred_decision, dtype=bool)
        mask[np.arange(y_true.shape[0]), y_true] = False
        margin = pred_decision[~mask]
        margin -= np.max(pred_decision[mask].reshape(y_true.shape[0], -1), axis=1)

    else:
        # Handles binary class case
        # this code assumes that positive and negative labels
        # are encoded as +1 and -1 respectively
        pred_decision = column_or_1d(pred_decision)
        pred_decision = np.ravel(pred_decision)

        lbin = LabelBinarizer(neg_label=-1)
        y_true = lbin.fit_transform(y_true)[:, 0]

        try:
            margin = y_true * pred_decision
        except TypeError:
            raise TypeError("pred_decision should be an array of floats.")

    losses = 1 - margin
    # The hinge_loss doesn't penalize good enough predictions.
    np.clip(losses, 0, None, out=losses)
    return losses