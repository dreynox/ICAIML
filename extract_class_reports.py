import json

with open("ICAIML.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell["cell_type"] == "code":
        source = "".join(cell.get("source", []))
        if "classification_report" in source or "print" in source:
            for out in cell.get("outputs", []):
                if out.get("output_type") == "stream":
                    text = "".join(out.get("text", []))
                    if "precision" in text and "recall" in text:
                        print("==== FOUND REPORT ====")
                        print(source[:50])
                        print(text)
                        
                elif out.get("output_type") == "display_data":
                     data = out.get("data", {})
                     if "text/plain" in data:
                         text = "".join(data["text/plain"])
                         if "precision" in text and "recall" in text:
                             print(text)
