import os
import re
import pathlib


def replace(match):
    try:
        # Prefer using ints if the f and h values don't have a decimal point:
        return rf"f: {match.group(1)}, h: {match.group(2)}, g: {int(match.group(1)) - int(match.group(2))}"
    except Exception:
        return rf"f: {match.group(1)}, h: {match.group(2)}, g: {float(match.group(1)) - float(match.group(2))}"


pattern = re.compile(r"f: ([\d.]+), h: ([\d.]+), g: [\d.]+")  # f: 28, h: 28, g: 0

for dirpath, dirnames, filenames in os.walk("."):
    dirpath = pathlib.Path(dirpath)
    dirname = dirpath.name
    if not dirname.startswith("nodes-"):
        continue
    for filename in filenames:
        if filename.endswith("-fixed"):
            continue
        filepath = dirpath.joinpath(filename)
        fixedpath = dirpath.joinpath(filename + "-fixed")
        print(f"Working on {filepath}")
        with open(os.path.join(dirpath, filename)) as f, open(fixedpath, "w") as f_fixed:
            for line in f:
                try:
                    line = pattern.sub(replace, line)
                    f_fixed.write(line)
                except Exception as e:
                    f_fixed.write(line)
        fixedpath.replace(filepath)
