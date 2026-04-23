import json
import pandas as pd
from IPython.display import display

with open("ICAIML.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("----- TEXT OUTPUTS -----")
for cell in nb.get("cells", []):
    if cell["cell_type"] == "code":
        outputs = cell.get("outputs", [])
        for out in outputs:
            if out.get("output_type") == "stream" and out.get("name") == "stdout":
                text = "".join(out.get("text", []))
                if "Accuracy:" in text or "ROC" in text or "CV Accuracy" in text or "Drop" in text:
                    pass # We will grab pandas html outputs mostly
            elif out.get("output_type") in ["execute_result", "display_data"]:
                data = out.get("data", {})
                if "text/plain" in data:
                    text = "".join(data["text/plain"])
                    if "Accuracy" in text and "Model" in text:
                        print("==== FOUND METRICS TABLE ====")
                        print(text)
                if "text/html" in data:
                    text = "".join(data["text/html"])
                    if "Accuracy" in text or "Model" in text or "CV" in text:
                        # try to parse as df?
                        pass

print("----- TEXT SEARCH FOR METRICS -----")
for cell in nb.get("cells", []):
    if cell["cell_type"] == "code":
        source = "".join(cell.get("source", []))
        if "evaluation_summary_df =" in source or "cv_df =" in source or "noise_results" in source:
             for out in cell.get("outputs", []):
                 if "data" in out and "text/plain" in out["data"]:
                     print(out["data"]["text/plain"])
