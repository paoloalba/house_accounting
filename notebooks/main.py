import os
import sys

sys.path.append(os.getenv("GLOBAL_LIBRARIES_PATH", ""))
sys.path = list(set(sys.path))

pmt_storage = os.getenv("PMT_STG_PATH")

print()