"""Structured errors: every failure has a stable code, a message and details.

Why: the evaluator must never crash on one bad file. It catches DataLoadError
and writes {"code": ..., "message": ...} into the results instead.
"""


class DataLoadError(Exception):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self):
        return {"code": self.code, "message": self.message, "details": self.details}
