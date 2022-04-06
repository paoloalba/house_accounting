import os
import yaml

with open(os.path.join(os.getenv("PMT_STG_PATH"), "config.yaml"), "r") as f:
    global_config = yaml.safe_load(f)
