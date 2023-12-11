import os
import json

def save_to_key(path, key, value):
    if not os.path.exists(path):
        with open(path, 'w') as fp:
            json.dump({}, fp)

    with open(path, 'r') as fp:
            dict = json.load(fp)
    
    dict[key] = value

    with open(path, 'w') as fp:
        json.dump(dict, fp)
