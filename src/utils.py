import os
import pickle
from pathlib import Path

def pickled_resource(pickle_path:str, generation_func:callable, *args, **kwargs):
    if pickle_path is None:
        return generation_func(*args, **kwargs)
    else:
        if os.path.exists(pickle_path):
            return pickle.load(open(pickle_path, 'rb'))
        else:
            instance = generation_func(*args, **kwargs)
            os.makedirs(str(Path(pickle_path).parent), exist_ok=True)
            pickle.dump(instance, open(pickle_path, 'wb'), pickle.HIGHEST_PROTOCOL)
            return instance