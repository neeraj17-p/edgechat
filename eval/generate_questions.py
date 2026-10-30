"""
generate_questions.py
──────────────────────
Builds eval/questions.json with 35 benchmark questions (27 in-domain, 8 unanswerable/out-of-domain)
grounded strictly in the COEP Technological University documents.
"""

import json
from pathlib import Path

QUESTIONS = [
    # ── Category 1: Academic Regulations & Attendance ──
    {
        "q_id": "q01",
        "category": "Academic Regulations",
        "question": "What is the minimum attendance required for students to be permitted to attend the Mid Semester Examination (MSE)?",
        "gold_answer": "Each student is required to attend at least 75% of all conducted classes (lectures, tutorials, laboratories, studios, workshops) to be permitted to attend the Mid Semester Examination.",
        "gold_document": "UG-Programmes-Academic-Rules.pdf",
        "gold_page": 21,
        "answerable_in_docs": True,
    },
    {
        "q_id": "q02",
        "category": "Academic Regulations",
        "question": "What type of grading policy is used for courses according to the UG Academic Rules?",
        "gold_answer": "The grading is relative and designed to approximate a normal distribution for a particular course, based on the statistical distribution of grand totals obtained by students.",
        "gold_document": "UG-Programmes-Academic-Rules.pdf",
        "gold_page": 22,
        "answerable_in_docs": True,
    },
    {
        "q_id": "q03",
        "category": "Academic Regulations",
        "question": "Whenever a student repeats or substitutes a course, which of the two grades is ignored in the computation of CGPA?",
        "gold_answer": "The lower of the two grades obtained by the student in the course shall be ignored in the computation of CGPA from that semester onwards.",
        "gold_document": "UG-Programmes-Academic-Rules.pdf",
        "gold_page": 23,
        "answerable_in_docs": True,
    },

    # ── Category 2: Examination Code of Conduct & Unfair Means ──
    {
        "q_id": "q04",
        "category": "Examination Rules",
        "question": "Are wrist watches allowed during examinations according to the Code of Conduct for students?",
        "gold_answer": "Strictly, wrist watches of any type (smart, digital, programmable, analog, or any other) are not allowed during examinations.",
        "gold_document": "COEP_Code-of-Conduct-during-Examination-for-Students-2024.pdf",
        "gold_page": 1,
        "answerable_in_docs": True,
    },
    {
        "q_id": "q05",
        "category": "Examination Rules",
        "question": "What happens if a wrist watch, jacket, or cap is found with a student during an examination?",
        "gold_answer": "It will be confiscated and returned after the last day of scheduled examination in the timetable.",
        "gold_document": "COEP_Code-of-Conduct-during-Examination-for-Students-2024.pdf",
        "gold_page": 1,
        "answerable_in_docs": True,
    },
    {
        "q_id": "q06",
        "category": "Examination Rules",
        "question": "What type of footwear is permitted in the examination hall according to the student code of conduct?",
        "gold_answer": "Shoes of any type are not allowed; only sandals or slippers are permitted.",
        "gold_document": "COEP_Code-of-Conduct-during-Examination-for-Students-2024.pdf",
        "gold_page": 1,
        "answerable_in_docs": True,
    },
    {
        "q_id": "q07",
        "category": "Examination Rules",
        "question": "What envelope color is used to seal the Answer Book of a student observed using unfair means during an exam?",
        "gold_answer": "The Answer Book and concerned belongings are sealed immediately in a green envelope signed by the student and invigilator/squad.",
        "gold_document": "COEP_Final-Guidelines-for-Unfair-Means-2025.pdf",
        "gold_page": 1,
        "answerable_in_docs": True,
    },
    {
        "q_id": "q08",
        "category": "Examination Rules",
        "question": "What is the penalty if a student is caught talking to another student inside the examination hall without permission?",
        "gold_answer": "The punishment is discontinuation from the examination of that particular course, and the grade in that course is awarded based on performance up to that point.",
        "gold_document": "COEP_Final-Guidelines-for-Unfair-Means-2025.pdf",
        "gold_page": 3,
        "answerable_in_docs": True,
    },

    # ── Category 3: Hostel Rules & Anti-Ragging ──
    {
        "q_id": "q09",
        "category": "Hostel Regulations",
        "question": "What act prohibits ragging according to the COEP hostel rules and regulations?",
        "gold_answer": "The Maharashtra Prohibition of Ragging Act, 1999 (Maharashtra Act No. XXXIII of 1999), passed on 7th April 1999, prohibits ragging.",
        "gold_document": "Hostel-Rules-and-Regulation._COEP-Tech.-2025.pdf",
        "gold_page": 1,
        "answerable_in_docs": True,
    },
    {
        "q_id": "q10",
        "category": "Hostel Regulations",
        "question": "Is ragging considered a criminal offence under the COEP hostel regulations?",
        "gold_answer": "Yes, ragging is strictly prohibited and is a cognizable criminal offence under the Maharashtra Prohibition of Ragging Act, 1999.",
        "gold_document": "Hostel-Rules-and-Regulation._COEP-Tech.-2025.pdf",
        "gold_page": 1,
        "answerable_in_docs": True,
    },

    # ── Category 4: University Administration & Leadership ──
    {
        "q_id": "q11",
        "category": "Administration",
        "question": "Who is the Chairman of the Board of Governance of COEP Technological University?",
        "gold_answer": "Shri. Vinayak Pai, Managing Director & CEO of TATA Projects Limited, Mumbai.",
        "gold_document": "Board of Governance  COEP Technological University (COEP Tech).pdf",
        "gold_page": 1,
        "answerable_in_docs": True,
    },
    {
        "q_id": "q12",
        "category": "Administration",
        "question": "Who is the Vice-Chancellor of COEP Tech University according to the official documents?",
        "gold_answer": "Prof. Sunil G. Bhirud is the Vice-Chancellor of COEP Technological University.",
        "gold_document": "Board of Governance  COEP Technological University (COEP Tech).pdf",
        "gold_page": 1,
        "answerable_in_docs": True,
    },
    {
        "q_id": "q13",
        "category": "Administration",
        "question": "Who has been appointed as the Ombudsperson for Students at COEP Tech University?",
        "gold_answer": "Prof. Nitin Karmalkar, Former Vice-Chancellor of Savitribai Phule Pune University.",
        "gold_document": "Ombudsperson for Students  COEP Technological University (COEP Tech).pdf",
        "gold_page": 1,
        "answerable_in_docs": True,
    },

    # ── Category 5: PhD Rules & Regulations ──
    {
        "q_id": "q14",
        "category": "PhD Regulations",
        "question": "What is the minimum grade required in each Ph.D. coursework course for successful completion?",
        "gold_answer": "Ph.D. candidates must complete the coursework with a minimum grade of 'CC' in each of the courses on the 10-point grading scale.",
        "gold_document": "Guidelines-Rules-and-Regulations-Governing-Ph.D-Programs.pdf",
        "gold_page": 22,
        "answerable_in_docs": True,
    },
    {
        "q_id": "q15",
        "category": "PhD Regulations",
        "question": "Within what time frame must the entire evaluation process of a Ph.D. thesis be completed?",
        "gold_answer": "The entire evaluation process of a Ph.D. thesis shall be completed within a period of six months from the date of submission.",
        "gold_document": "Guidelines-Rules-and-Regulations-Governing-Ph.D-Programs.pdf",
        "gold_page": 34,
        "answerable_in_docs": True,
    },

    # ── Category 6: Academic Calendar & Dates ──
    {
        "q_id": "q16",
        "category": "Academic Calendar",
        "question": "What are the dates for the Student Induction Programme for First Year B. Tech and B. Planning for Term-I AY 2026-27?",
        "gold_answer": "The Student Induction Programme is scheduled from 21st September 2026 (Monday) to 24th September 2026 (Thursday).",
        "gold_document": "Academic Calendar FY B Tech & B Planning AY 2026-27.pdf",
        "gold_page": 1,
        "answerable_in_docs": True,
    },
    {
        "q_id": "q17",
        "category": "Academic Calendar",
        "question": "When does instruction begin for Even Semester (Term-II) for FY B. Tech and B. Planning according to the 2026-27 calendar?",
        "gold_answer": "Instruction begins on 11th January 2027 (Monday).",
        "gold_document": "Academic Calendar FY B Tech & B Planning AY 2026-27.pdf",
        "gold_page": 2,
        "answerable_in_docs": True,
    },
    {
        "q_id": "q18",
        "category": "Academic Calendar",
        "question": "When does semester instruction begin for 3rd, 5th, and 7th semester UG students for Term-I AY 2026-27?",
        "gold_answer": "Instruction begins on 6th July 2026 (Monday).",
        "gold_document": "Academic Calendar for UG SY, TY,  Final Year B-TECH and SY M-TECH 2026-27.pdf",
        "gold_page": 1,
        "answerable_in_docs": True,
    },

    # ── Category 7: Schemes & Scholarships ──
    {
        "q_id": "q19",
        "category": "Financial Schemes",
        "question": "Which Union Budget announced financial support under the PM Vidyalaxmi Scheme for education loans?",
        "gold_answer": "The Union Budget 2024-25 announced financial support for education loans under the PM Vidyalaxmi Scheme.",
        "gold_document": "PM_Vidyalaxmi_Scheme_Guidelines.pdf",
        "gold_page": 1,
        "answerable_in_docs": True,
    },

    # ── Category 8: Placements & Higher Studies ──
    {
        "q_id": "q20",
        "category": "Placements",
        "question": "Which department's B.Tech placement and higher studies report is documented for the year 2021-22 in the placement files?",
        "gold_answer": "The Department of Electronics and Telecommunication (B.Tech Placement and Higher Studies off-campus 2021-22).",
        "gold_document": "Placements__Higher-Studies_2021-22.pdf",
        "gold_page": 1,
        "answerable_in_docs": True,
    },

    # ── Category 9: Laboratories & Facilities ──
    {
        "q_id": "q21",
        "category": "Laboratories",
        "question": "What is the reported cost and number of CROs in the Electronics Laboratory of the Electronics & Telecommunication Department?",
        "gold_answer": "The reported cost is 5.16 Lakhs and it contains 15 CROs.",
        "gold_document": "Laboratories-Major-Equipments.pdf",
        "gold_page": 2,
        "answerable_in_docs": True,
    },
    {
        "q_id": "q22",
        "category": "Laboratories",
        "question": "What is the cost of the Digital Signal Processing Laboratory in the Electronics & Telecommunication Department?",
        "gold_answer": "The cost of the Digital Signal Processing Laboratory is 4.20 Lakhs.",
        "gold_document": "Laboratories-Major-Equipments.pdf",
        "gold_page": 3,
        "answerable_in_docs": True,
    },

    # ── Category 10: PG Academic Rules ──
    {
        "q_id": "q23",
        "category": "PG Regulations",
        "question": "How many times is the End Semester Examination (ESE) conducted for a course in the main semester for PG programmes?",
        "gold_answer": "ESE shall be conducted once for a course in the main semester for undergoing PG programmes.",
        "gold_document": "PG-Programmes-Academic-Rules-and-Regulations-AY-2025-26-Modified-06072026-.pdf",
        "gold_page": 19,
        "answerable_in_docs": True,
    },

    # ── Category 11: History & Heritage ──
    {
        "q_id": "q24",
        "category": "University History",
        "question": "In what year was the College of Engineering Pune (COEP) established according to the official documents?",
        "gold_answer": "COEP was established in the year 1854.",
        "gold_document": "Laboratories-Major-Equipments.pdf",
        "gold_page": 1,
        "answerable_in_docs": True,
    },

    # ── Category 12: Additional Academic Regulations ──
    {
        "q_id": "q25",
        "category": "Academic Regulations",
        "question": "What transitional letter grade is used by instructors as described in the UG Academic Rules?",
        "gold_answer": "The transitional letter grade 'IC' (Incomplete Course) is used by instructors.",
        "gold_document": "UG-Programmes-Academic-Rules.pdf",
        "gold_page": 22,
        "answerable_in_docs": True,
    },
    {
        "q_id": "q26",
        "category": "Examination Rules",
        "question": "What disciplinary consequence does impersonation in examinations lead to according to the Code of Conduct?",
        "gold_answer": "Impersonation leads to detention in the said subject with no re-exam, and disciplinary action against the student.",
        "gold_document": "COEP_Code-of-Conduct-during-Examination-for-Students-2024.pdf",
        "gold_page": 1,
        "answerable_in_docs": True,
    },
    {
        "q_id": "q27",
        "category": "Administration",
        "question": "Which former university did Prof. Nitin Karmalkar serve as Vice-Chancellor before becoming Ombudsperson at COEP Tech?",
        "gold_answer": "Prof. Nitin Karmalkar formerly served as the Vice-Chancellor of Savitribai Phule Pune University.",
        "gold_document": "Ombudsperson for Students  COEP Technological University (COEP Tech).pdf",
        "gold_page": 1,
        "answerable_in_docs": True,
    },

    # ── Category 13: Unanswerable & Out-of-Domain Questions (Must Refuse) ──
    {
        "q_id": "q28",
        "category": "Unanswerable (Out-of-Domain)",
        "question": "What is the fee structure and room rent for staying in the undergraduate hostels at IIT Bombay?",
        "gold_answer": "I could not find this in the provided documents.",
        "gold_document": None,
        "gold_page": None,
        "answerable_in_docs": False,
    },
    {
        "q_id": "q29",
        "category": "Unanswerable (Out-of-Domain)",
        "question": "Who was the commander of the Apollo 11 moon landing mission in 1969?",
        "gold_answer": "I could not find this in the provided documents.",
        "gold_document": None,
        "gold_page": None,
        "answerable_in_docs": False,
    },
    {
        "q_id": "q30",
        "category": "Unanswerable (Out-of-Domain)",
        "question": "What is the depth and water capacity of the Olympic-sized swimming pool at COEP Tech University?",
        "gold_answer": "I could not find this in the provided documents.",
        "gold_document": None,
        "gold_page": None,
        "answerable_in_docs": False,
    },
    {
        "q_id": "q31",
        "category": "Unanswerable (Out-of-Domain)",
        "question": "What are the rules and regulations for obtaining a flying drone pilot license in the United Kingdom?",
        "gold_answer": "I could not find this in the provided documents.",
        "gold_document": None,
        "gold_page": None,
        "answerable_in_docs": False,
    },
    {
        "q_id": "q32",
        "category": "Unanswerable (Out-of-Domain)",
        "question": "What is the exact mess fee refund policy for day-scholar students who skip lunch on Tuesdays?",
        "gold_answer": "I could not find this in the provided documents.",
        "gold_document": None,
        "gold_page": None,
        "answerable_in_docs": False,
    },
    {
        "q_id": "q33",
        "category": "Unanswerable (Out-of-Domain)",
        "question": "How many submarines does the Indian Navy currently operate in the Arabian Sea?",
        "gold_answer": "I could not find this in the provided documents.",
        "gold_document": None,
        "gold_page": None,
        "answerable_in_docs": False,
    },
    {
        "q_id": "q34",
        "category": "Unanswerable (Out-of-Domain)",
        "question": "What is the minimum GPA required to transfer into the MIT Media Lab from Stanford University?",
        "gold_answer": "I could not find this in the provided documents.",
        "gold_document": None,
        "gold_page": None,
        "answerable_in_docs": False,
    },
    {
        "q_id": "q35",
        "category": "Unanswerable (Out-of-Domain)",
        "question": "What are the syllabus topics for the Advanced Quantum Teleportation course at COEP?",
        "gold_answer": "I could not find this in the provided documents.",
        "gold_document": None,
        "gold_page": None,
        "answerable_in_docs": False,
    },
]

if __name__ == "__main__":
    out_file = Path("eval/questions.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(QUESTIONS, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(QUESTIONS)} questions in {out_file} (27 in-domain, 8 unanswerable).")
