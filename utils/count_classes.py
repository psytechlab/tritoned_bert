import sys
import json

with open(sys.argv[1]) as f:
    print(len(json.load(f)))