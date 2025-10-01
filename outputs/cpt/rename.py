import os
import re

# Work in current directory
for fname in os.listdir("."):
    if fname.endswith(".pkl") and os.path.isfile(fname):
        # Remove extension
        base, ext = os.path.splitext(fname)

        # Regex: keep everything up to the last digit sequence
        # and drop anything that follows before the extension
        match = re.search(r"^(.*\d)(?:\D.*)?$", base)
        if match:
            new_base = match.group(1)
            new_name = new_base + ext
            if new_name != fname:
                print(f"Renaming: {fname} -> {new_name}")
                os.rename(fname, new_name)