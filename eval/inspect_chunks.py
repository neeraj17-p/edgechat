"""
Find representative questions and exact gold passages across COEP documents.
"""
import json
from pathlib import Path

chunks_file = Path("data/index/chunks.json")
chunks = json.load(open(chunks_file, encoding="utf-8"))

keywords = [
    ("Vice-Chancellor", "vice chancellor"),
    ("Attendance", "attendance"),
    ("Wrist Watch", "wrist watch"),
    ("Ragging Act", "ragging"),
    ("CGPA / Backlog", "cgpa"),
    ("Hostel Fees", "mess"),
    ("Placement", "highest package"),
    ("Unfair Means", "malpractice"),
    ("PM Vidyalaxmi", "interest subvention"),
    ("PhD Coursework", "comprehensive examination"),
    ("Academic Calendar", "mid semester examination"),
    ("Ombudsperson", "ombudsperson"),
    ("Branch Change", "change of branch"),
    ("Grading Scale", "grade points"),
]

print(f"Total chunks loaded: {len(chunks)}")
for label, kw in keywords:
    matches = [c for c in chunks if kw in c["text"].lower()]
    print(f"=== {label} (Keyword: '{kw}') -> {len(matches)} matches ===")
    if matches:
        first = matches[0]
        print(f"Doc: {first['doc_name']} | Page: {first['page_number']}")
        snippet = " ".join(first['text'].split()[:50])
        print(f"Snippet: {snippet}...\n")
