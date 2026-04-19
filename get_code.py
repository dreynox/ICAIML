import json

with open("ICAIML.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("code.py", "w", encoding="utf-8") as f_out:
    for i, cell in enumerate(nb.get("cells", [])):
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            f_out.write(f"# ----- Cell {i} -----\n")
            f_out.write(source + "\n\n")
