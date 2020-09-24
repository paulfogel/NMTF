import numpy as np
import json
from datetime import date, datetime


class JSONEncoder(json.JSONEncoder):
    """
    Extending the JSON encoder so it knows how to serialise a dataframe
    """

    def default(self, obj):
        if hasattr(obj, "to_json"):
            if callable(obj.to_json):
                try:
                    return obj.to_json(orient="records")
                except TypeError:
                    return obj.to_json()
            else:
                return obj.to_json
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, date):
            return obj.strftime("%Y-%x-%d")
        elif isinstance(obj, datetime):
            return obj.strftime("%Y-%x-%d %h:%M:%S")
        else:
            return json.JSONEncoder.default(self, obj)