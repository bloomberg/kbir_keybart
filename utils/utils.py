import json
import os


def update_tokenizer_config(dirname, filename):
    if os.path.exists(os.path.join(dirname, filename)):
        with open(os.path.join(dirname, filename), "r") as f:
            for line in f:
                data = json.loads(line)
                data["tokenizer_file"] = os.path.join(dirname, "tokenizer.json")
                break
        with open(os.path.join(dirname, filename), "w") as f:
            f.write(json.dumps(data))
