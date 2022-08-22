from typing import List, Tuple, Union


class DataTracker:
    def __init__(self, value_range: Union[Tuple, List], mode="greater", extra_data=None) -> None:
        assert mode in ["greater", "lesser"], f"Unkown mode '{mode}'"
        assert (
            isinstance(value_range, (tuple, list)) and value_range[0] < value_range[1]
        ), f"Value range '{value_range}' must be ordered from smalest to highest"

        self.mode = mode
        self.value_range = value_range

        self.best_val = value_range[0] if mode == "greater" else value_range[1]
        self.extra_data = extra_data

    def update(self, val, extra_data=None):
        if self.mode == "greater":
            self.update_greater(val, extra_data=extra_data)
        elif self.mode == "lesser":
            self.update_lesser(val, extra_data=extra_data)

    def update_greater(self, val, extra_data=None):
        if val >= self.best_val:
            self.best_val = val
            if extra_data:
                self.extra_data = extra_data

    def update_lesser(self, val, extra_data=None):
        if val < self.best_val:
            self.best_val = val
            if extra_data:
                self.extra_data = extra_data

    def get_best_value(self):
        return self.best_val

    def get_extra_data(self):
        return self.extra_data
