import os
import json

class utils:
    def __init__(self):
        pass

    def dircheck(self, path):
        os.makedirs(path, exist_ok=True)

    def mdm(self, config_path):
        with open(config_path) as file:
                config = json.load(file)
        return config
