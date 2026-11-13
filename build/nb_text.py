"""Print the text outputs of an executed notebook, and optionally save its images.

    python build/nb_text.py <notebook.ipynb> [--img <dir>] [--max <chars per output>] [--cells 3,5-9]

Use it to read a Kaggle output notebook without opening a 2 MB JSON file.
"""
import base64
import json
import os
import sys


def _cells_arg(text):
    keep = set()
    for part in text.split(","):
        a, _, b = part.partition("-")
        keep.update(range(int(a), int(b or a) + 1))
    return keep


def main(argv):
    path = argv[1]
    img_dir = argv[argv.index("--img") + 1] if "--img" in argv else None
    max_chars = int(argv[argv.index("--max") + 1]) if "--max" in argv else 4000
    keep = _cells_arg(argv[argv.index("--cells") + 1]) if "--cells" in argv else None
    sys.stdout.reconfigure(encoding="utf-8")
    nb = json.load(open(path, encoding="utf-8"))
    if img_dir:
        os.makedirs(img_dir, exist_ok=True)
    for i, cell in enumerate(nb["cells"]):
        if keep is not None and i not in keep:
            continue
        src = "".join(cell["source"])
        head = src.strip().splitlines()[0][:100] if src.strip() else ""
        print(f"\n=== cell {i} [{cell['cell_type']}] {head}")
        if cell["cell_type"] != "code":
            continue
        for k, out in enumerate(cell.get("outputs", [])):
            kind = out["output_type"]
            if kind == "stream":
                text = "".join(out["text"])
            elif kind == "error":
                text = "ERROR " + out.get("ename", "") + ": " + out.get("evalue", "")
            else:
                data = out.get("data", {})
                text = "".join(data.get("text/plain", ""))
                if "image/png" in data:
                    text = "[image] " + text
                    if img_dir:
                        name = os.path.join(img_dir, f"cell{i:02d}_{k}.png")
                        with open(name, "wb") as f:
                            f.write(base64.b64decode(data["image/png"]))
            if len(text) > max_chars:
                text = text[: max_chars // 2] + "\n   [...]\n" + text[-max_chars // 2:]
            print(text.rstrip())


if __name__ == "__main__":
    main(sys.argv)
