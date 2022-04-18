import os
import yaml

try:
    with open(os.path.join(os.getenv("PMT_STG_PATH"), "config.yaml"), "r") as f:
        global_config = yaml.safe_load(f)
except:
    global_config = {"example_db_name": "example_cashflows.db"}
