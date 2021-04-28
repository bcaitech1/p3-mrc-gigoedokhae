import re
from glob import glob
from pathlib import Path

def increment_path(path, infix="", name=""):
    path += infix
    dirs = glob(f"{path}*")
    stem = Path(path).stem
    matches = [re.search(rf"{stem}(\d+)", d) for d in dirs]
    i = [int(m.groups()[0]) for m in matches if m]
    n = max(i) + 1 if i else 1
    return "".join([path, f"{n}_", name])