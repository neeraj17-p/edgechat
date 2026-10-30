"""
Extract exact text snippets for building ground truth questions.
"""
import json
import re

chunks = json.load(open("data/index/chunks.json", encoding="utf-8"))

def inspect_doc(doc_substr, filter_kw=None, max_results=3):
    print(f"\n--- DOC: '{doc_substr}' (Filter: {filter_kw}) ---")
    count = 0
    for c in chunks:
        if doc_substr.lower() in c["doc_name"].lower():
            if filter_kw is None or filter_kw.lower() in c["text"].lower():
                safe_text = re.sub(r'[^\x00-\x7F]+', ' ', c['text'])
                print(f"[{c['doc_name']} | Page {c['page_number']}]")
                lines = [line.strip() for line in safe_text.split('\n') if line.strip()][:8]
                print("\n".join(lines))
                print("---")
                count += 1
                if count >= max_results:
                    break

inspect_doc("Vice Chancellor", "email")
inspect_doc("Chancellor", "Governor")
inspect_doc("Ombudsperson")
inspect_doc("UG-Programmes-Academic-Rules", "attendance")
inspect_doc("UG-Programmes-Academic-Rules", "grading")
inspect_doc("COEP_Final-Guidelines-for-Unfair-Means", "committee")
inspect_doc("Academic Calendar FY", "term")
inspect_doc("Hostel-Rules", "fees")
inspect_doc("Placements", "company")
inspect_doc("Laboratories-Major-Equipments", "lab")
