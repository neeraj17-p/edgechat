"""
Search specific chunks and print exact document name, page number, and text.
"""
import json
import re

chunks = json.load(open("data/index/chunks.json", encoding="utf-8"))

def search(query, limit=2):
    print(f"\n==================== SEARCH: '{query}' ====================")
    q_words = [w.lower() for w in query.split()]
    scored = []
    for c in chunks:
        text_lower = c["text"].lower()
        score = sum(1 for w in q_words if w in text_lower)
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    for s, c in scored[:limit]:
        safe_text = re.sub(r'[^\x00-\x7F]+', ' ', c['text'])
        print(f"[{c['doc_name']} | Page {c['page_number']}] (match count: {s})")
        print(f"{safe_text[:350]}...\n")

queries = [
    "attendance mid semester examination",
    "wrist watch allowed code of conduct",
    "maharashtra prohibition of ragging act",
    "vice chancellor email address coep",
    "ombudsperson for students coep",
    "unfair means mobile phone exam penalty",
    "highest package placement 2021 22",
    "phd comprehensive examination credit coursework",
    "grade point 10 point scale o a b",
    "pm vidyalaxmi scheme loan amount interest",
    "change of branch eligibility cgpa",
    "hostel curfew timings gates closed",
    "provisional degree certificate passing",
    "re-examination grade cap academic rules",
    "board of governance chairman name",
]

for q in queries:
    search(q, limit=1)
