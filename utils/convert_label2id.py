import sys
import json

with open(sys.argv[1]) as f:
    label2id = json.load(f)

try:
    id2label = {int(v):k for k,v in label2id.items()}
except ValueError:
    print("Probably a file is already id2label or it's incorrect.")
    sys.exit(1)

if len(sys.argv) < 3:
    out_path = sys.argv[1].split("/")
    out_path = "/".join(out_path[:-1] + ["id2label.json"])
else:
    out_path = sys.argv[2]
    
with open(out_path, 'w') as f:
    json.dump(id2label, f, ensure_ascii=False, indent=4)
