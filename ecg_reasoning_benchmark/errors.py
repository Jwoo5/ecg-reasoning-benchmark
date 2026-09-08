"""Exceptions the inference loop treats specially."""


class UnusableModelOutput(Exception):
    """A model produced output that cannot answer the current step.

    Raised by a model adapter that can tell its own bad output apart from a
    request that never ran. :meth:`Inferencer.proceed_step` records it against
    the step and goes on to the next one, which is meaningful because every
    reasoning step teacher-forces the ground truth into the history rather than
    the model's own answer: the next step sees exactly the history it would
    have seen had this one succeeded.
    """
