import os
import pickle
from pathlib import Path


# pickle the output of a function
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


# sort the documents (with labels) based on length
def sort_docs(docs, labels):
    tuples = list(zip(docs, labels))
    tuples = sorted(tuples, key=lambda x: len(x[0]))
    return list(zip(*tuples))


# return max doc length in a series of documents
def docs_max_len(docs):
    return len(max(docs, key=len))

