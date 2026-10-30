"""
create_sample_pdfs.py
─────────────────────
One-time script to generate 5 realistic sample PDFs for testing.
These mimic real college documents: attendance policy, exam rules,
academic regulations, syllabus, and placement policy.

Run once from the project root:
    .\\venv\\Scripts\\python.exe create_sample_pdfs.py

After this, replace with your real PDFs in data/docs/.
"""

from pathlib import Path
import fitz  # PyMuPDF


OUTPUT_DIR = Path("data/docs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def new_pdf() -> fitz.Document:
    return fitz.open()


def add_page(doc: fitz.Document, title: str, body_lines: list[str]) -> None:
    """Add one page to a fitz Document with a title and body text."""
    page = doc.new_page(width=595, height=842)  # A4 in points

    # Title
    page.insert_text(
        (50, 60),
        title,
        fontsize=16,
        fontname="helv",
        color=(0, 0, 0),
    )

    # Horizontal rule simulation
    page.draw_line((50, 75), (545, 75), color=(0, 0, 0), width=0.5)

    # Body text
    y = 100
    for line in body_lines:
        if line == "":
            y += 9
            continue
        while len(line) > 90:
            page.insert_text((50, y), line[:90], fontsize=10, fontname="helv")
            line = "    " + line[90:]
            y += 14
        page.insert_text((50, y), line, fontsize=10, fontname="helv")
        y += 14
        if y > 800:
            break


def save(doc: fitz.Document, name: str) -> None:
    path = OUTPUT_DIR / name
    doc.save(str(path))
    doc.close()
    print(f"  Created: {path}  ({path.stat().st_size // 1024} KB)")


# ── Document 1: Attendance Policy ────────────────────────────────────────────

def create_attendance_policy():
    doc = new_pdf()

    add_page(doc, "ATTENDANCE POLICY -- Academic Year 2024-25", [
        "1. MINIMUM ATTENDANCE REQUIREMENT",
        "",
        "All students enrolled in undergraduate programs at Greenfield College of",
        "Engineering and Technology must maintain a minimum attendance of 75% in",
        "each subject per semester. This requirement applies to theory classes,",
        "laboratory sessions, and tutorials separately.",
        "",
        "Students who fail to meet the 75% threshold in any subject will be",
        "declared DETAINED and will NOT be permitted to appear in the End",
        "Semester Examination (ESE) for that subject.",
        "",
        "2. CONDONATION OF ATTENDANCE",
        "",
        "Students with attendance between 65% and 74% may apply for condonation",
        "of shortage through the Dean of Academics within the first week of the",
        "examination schedule. Condonation is granted only on medical grounds or",
        "participation in university-approved inter-collegiate events.",
        "",
        "Supporting documents (medical certificate from a registered physician,",
        "or event participation certificate) must be submitted along with the",
        "condonation application. The decision of the Dean is final.",
        "",
        "3. ATTENDANCE BELOW 65%",
        "",
        "No condonation will be granted for attendance below 65% under any",
        "circumstances. Such students must re-enroll in the subject in a",
        "subsequent semester.",
    ])

    add_page(doc, "ATTENDANCE POLICY -- Page 2", [
        "4. CALCULATION OF ATTENDANCE",
        "",
        "Attendance is calculated as:",
        "   Attendance % = (Classes Attended / Total Classes Held) x 100",
        "",
        "Lab sessions count as 2 periods each for attendance calculation.",
        "Tutorial sessions count as 1 period each.",
        "",
        "5. REPORTING AND MONITORING",
        "",
        "Attendance is recorded by the respective subject faculty using the",
        "College Management System (CMS) at the beginning of each class.",
        "Students may view their real-time attendance on the student portal.",
        "",
        "Attendance reports are compiled at the end of each month and displayed",
        "on the department notice board. Students with attendance below 75%",
        "receive an automated SMS and email alert after the 30th, 60th, and",
        "90th day of each semester.",
        "",
        "6. LEAVE OF ABSENCE",
        "",
        "Students seeking planned leave must submit a Leave Application to the",
        "class advisor at least 3 working days in advance. Medical leave must",
        "be supported by a medical certificate within 2 days of returning.",
        "Unplanned absences beyond 3 consecutive days require a parent letter.",
        "",
        "7. SPECIAL CASES -- SPORTS AND CULTURAL EVENTS",
        "",
        "Students representing the college in inter-collegiate, state-level, or",
        "national-level events are marked OD (On Duty) for the duration of",
        "their participation. OD periods are counted as present for attendance",
        "calculation. Prior approval from the Head of Department is mandatory.",
    ])

    add_page(doc, "ATTENDANCE POLICY -- Page 3", [
        "8. DISCIPLINARY ACTION FOR PROXY ATTENDANCE",
        "",
        "Proxy attendance (marking another student present) is a serious",
        "disciplinary offence. Any student found involved in proxy attendance,",
        "whether as the requestor or the respondent, will face:",
        "   - Cancellation of attendance for the entire month",
        "   - A written warning placed on academic record",
        "   - Suspension of 3 to 10 working days for repeat offences",
        "   - Referral to the Disciplinary Committee for severe cases",
        "",
        "9. APPEALS",
        "",
        "Appeals regarding attendance discrepancies must be submitted in writing",
        "to the class advisor within 5 working days of the attendance report",
        "publication. The class advisor will review with the faculty concerned",
        "and respond within 7 working days.",
        "",
        "10. EFFECTIVE DATE",
        "",
        "This policy is effective from 1 August 2024 and supersedes all",
        "previous attendance circulars issued by Greenfield College.",
        "",
        "Approved by: Dr. Ramesh Iyer, Dean of Academics",
        "Date: 28 July 2024",
    ])

    save(doc, "attendance_policy.pdf")


# ── Document 2: Exam Rules ────────────────────────────────────────────────────

def create_exam_rules():
    doc = new_pdf()

    add_page(doc, "EXAMINATION RULES AND REGULATIONS -- 2024-25", [
        "1. EXAMINATION SCHEDULE",
        "",
        "The End Semester Examination (ESE) for all undergraduate programs",
        "will be conducted from 18 November 2024 (Monday) to 6 December 2024",
        "(Friday). The Internal Assessment Examination (IAE) will be held",
        "from 14 October 2024 to 19 October 2024.",
        "",
        "Hall tickets must be collected from the examination office between",
        "11 November 2024 and 15 November 2024. Students without a valid",
        "hall ticket will not be permitted to enter the examination hall.",
        "",
        "2. ELIGIBILITY TO APPEAR",
        "",
        "A student is eligible to appear in the ESE only if:",
        "   (a) Minimum 75% attendance is maintained in that subject",
        "   (b) All internal assessment components are completed",
        "   (c) All laboratory practical records are submitted and signed",
        "   (d) All dues to the college are cleared",
        "",
        "3. REPORTING TIME",
        "",
        "Students must report to the examination hall at least 15 minutes",
        "before the scheduled start time. Entry is not permitted after the",
        "first 30 minutes of the examination have elapsed.",
        "",
        "Students arriving late by up to 30 minutes may be allowed entry",
        "at the discretion of the Chief Superintendent but will not receive",
        "any extra time to compensate for the late arrival.",
    ])

    add_page(doc, "EXAMINATION RULES -- Page 2", [
        "4. PERMITTED AND PROHIBITED ITEMS",
        "",
        "Permitted inside the examination hall:",
        "   - Admit card / Hall ticket",
        "   - Valid College ID card",
        "   - Black or blue ballpoint pens (minimum 2)",
        "   - Pencil and eraser (for diagrams only)",
        "   - Scientific calculator (non-programmable) where applicable",
        "",
        "Strictly prohibited:",
        "   - Mobile phones and smartwatches (must be deposited outside)",
        "   - Programmable calculators or those with text memory",
        "   - Written notes, books, or any printed material",
        "   - Bluetooth earphones or any wireless communication device",
        "",
        "5. CONDUCT DURING EXAMINATION",
        "",
        "Students must maintain absolute silence throughout the examination.",
        "Communication between students in any form is strictly prohibited.",
        "Answer scripts must be written in English only, except for",
        "language-specific papers.",
        "",
        "Students must write their Register Number (not name) on the",
        "main answer booklet and every supplementary sheet. Using the",
        "name on the answer script may result in cancellation.",
        "",
        "6. MALPRACTICE AND UNFAIR MEANS",
        "",
        "Any student found using unfair means during the examination will",
        "have their answer script confiscated and a malpractice report filed.",
        "Penalties range from cancellation of the paper to debarment from",
        "all examinations for one full academic year, depending on severity.",
    ])

    add_page(doc, "EXAMINATION RULES -- Page 3", [
        "7. GRACE MARKS POLICY",
        "",
        "Grace marks of up to 5 marks per subject may be awarded to",
        "border-line students at the discretion of the examination committee,",
        "subject to university guidelines. Grace marks are applied ONLY to",
        "the ESE component and not to internal assessment.",
        "",
        "8. RE-EXAMINATION AND SUPPLEMENTARY EXAMS",
        "",
        "Students who fail in one or more subjects may appear in the",
        "Supplementary Examination conducted in April/May 2025.",
        "Registration for supplementary exams must be completed within",
        "15 days of the declaration of ESE results.",
        "",
        "9. RESULTS AND REVALUATION",
        "",
        "ESE results will be declared within 45 days of the last examination.",
        "Students may apply for photocopy of their answer script within",
        "10 days of result declaration. Revaluation applications must be",
        "submitted within 15 days of result declaration with a fee of",
        "Rs. 500 per subject.",
    ])

    save(doc, "exam_rules.pdf")


# ── Document 3: Academic Regulations ─────────────────────────────────────────

def create_academic_regulations():
    doc = new_pdf()

    add_page(doc, "ACADEMIC REGULATIONS -- B.E./B.Tech Programs (2022 Scheme)", [
        "1. PROGRAM STRUCTURE",
        "",
        "The B.E. / B.Tech program is of 4-year (8-semester) duration.",
        "Each semester is approximately 18 weeks long, including examinations.",
        "",
        "Credit Distribution per Semester (typical):",
        "   Theory subjects:     3-4 credits each (3-4 hours/week)",
        "   Laboratory subjects: 1-2 credits each (2-3 hours/week)",
        "   Mini project:        2 credits (Semester 5 and 6)",
        "   Main project:        6 credits (Semester 7 and 8 combined)",
        "",
        "Minimum total credits required for graduation: 160 credits",
        "",
        "2. GRADING SYSTEM",
        "",
        "The college follows an absolute grading system on a 10-point scale:",
        "",
        "   Marks Range    Grade    Grade Point",
        "   90 - 100       O        10",
        "   80 -  89       A+        9",
        "   70 -  79       A         8",
        "   60 -  69       B+        7",
        "   50 -  59       B         6",
        "   45 -  49       C         5",
        "   40 -  44       D         4",
        "   Below 40       F (Fail)  0",
        "",
        "The SGPA and CGPA are calculated at the end of each semester.",
    ])

    add_page(doc, "ACADEMIC REGULATIONS -- Page 2", [
        "3. PROMOTION RULES",
        "",
        "A student is promoted to the next year if:",
        "   - They have cleared all subjects of the previous year, OR",
        "   - They have a maximum of 2 backlogs (failed subjects) from",
        "     the previous year",
        "",
        "A student with 3 or more backlogs is NOT promoted and must",
        "re-enroll for the year. Maximum duration to complete the program",
        "is 7 years (2 times the normal 4-year duration minus 1).",
        "",
        "4. INTERNAL ASSESSMENT",
        "",
        "Each theory subject carries 100 marks total:",
        "   Internal Assessment (IA): 30 marks",
        "   End Semester Examination: 70 marks",
        "",
        "Internal Assessment components:",
        "   Test 1 (Unit 1 and 2):      10 marks",
        "   Test 2 (Unit 3 and 4):      10 marks",
        "   Assignment / Quiz / Viva:   10 marks",
        "",
        "A minimum of 14 marks (out of 30) in IA is required to appear",
        "in the ESE. Students scoring below 14 in IA are declared",
        "Not Qualified for the ESE of that subject.",
        "",
        "5. LABORATORY ASSESSMENT",
        "",
        "Each laboratory subject carries 100 marks:",
        "   Internal Continuous Assessment: 60 marks",
        "   End Semester Practical Exam:    40 marks",
    ])

    add_page(doc, "ACADEMIC REGULATIONS -- Page 3", [
        "6. PROJECT WORK GUIDELINES",
        "",
        "Semester 7 and 8 students must complete a Main Project in teams",
        "of 3 to 4 students. Solo projects are not permitted.",
        "The project must be approved by the Project Coordinator by the",
        "end of Week 3 of Semester 7.",
        "",
        "Project evaluation breakdown (200 marks total):",
        "   Phase 1 Review (Sem 7):    50 marks",
        "   Phase 2 Review (Sem 7):    50 marks",
        "   Final Viva Voce (Sem 8):   60 marks",
        "   Project Report:            40 marks",
        "",
        "7. HONORABLE MENTION AND DISTINCTIONS",
        "",
        "Students achieving a CGPA of 9.0 and above with no backlogs",
        "throughout the program will receive the First Class with",
        "Distinction certificate.",
        "",
        "The University Rank holder (top 3 students per branch) will",
        "receive a gold medal at the Annual Convocation.",
        "",
        "8. CODE OF CONDUCT",
        "",
        "Students are expected to maintain high academic integrity.",
        "Plagiarism in assignments or project reports will result in",
        "zero marks for that component. Repeat offences may result in",
        "disciplinary action as per the college conduct policy.",
    ])

    save(doc, "academic_regulations.pdf")


# ── Document 4: Syllabus ──────────────────────────────────────────────────────

def create_syllabus():
    doc = new_pdf()

    add_page(doc, "SYLLABUS -- CS301: Data Structures and Algorithms", [
        "Department: Computer Science and Engineering",
        "Semester: 3rd Semester (2022 Scheme)",
        "Credits: 4  |  Hours/Week: 3L + 1T  |  Marks: 70 (ESE) + 30 (IA)",
        "",
        "COURSE OBJECTIVES:",
        "1. Understand fundamental data structures and their operations",
        "2. Analyze the time and space complexity of algorithms",
        "3. Apply appropriate data structures to solve real problems",
        "4. Implement searching, sorting, and graph algorithms",
        "",
        "UNIT 1: Introduction and Arrays (9 hours)",
        "Algorithm analysis: time complexity, space complexity, Big-O notation.",
        "Arrays: one-dimensional, two-dimensional, multi-dimensional.",
        "Searching: linear search, binary search. Sorting: bubble, insertion,",
        "selection sort. Sparse matrices: representation and operations.",
        "",
        "UNIT 2: Linked Lists and Stacks (9 hours)",
        "Singly linked list: insertion, deletion, traversal, reversal.",
        "Doubly linked list and circular linked list operations.",
        "Stack: array and linked list representation, applications.",
        "Infix to postfix conversion, postfix evaluation. Tower of Hanoi.",
        "",
        "UNIT 3: Queues and Trees (9 hours)",
        "Queue: circular queue, priority queue, dequeue.",
        "Binary tree: representations, inorder/preorder/postorder traversals.",
        "Binary Search Tree: insertion, deletion, searching.",
        "AVL Trees: rotations, balancing. Heap: max-heap, min-heap, heap sort.",
    ])

    add_page(doc, "CS301 Syllabus -- Page 2", [
        "UNIT 4: Graphs (9 hours)",
        "Graph representation: adjacency matrix, adjacency list.",
        "Graph traversals: Breadth First Search (BFS), Depth First Search (DFS).",
        "Minimum Spanning Tree: Prim's algorithm, Kruskal's algorithm.",
        "Shortest path: Dijkstra's algorithm, Bellman-Ford algorithm.",
        "Topological sorting. Connected components.",
        "",
        "UNIT 5: Hashing and Advanced Sorting (9 hours)",
        "Hashing: hash functions, collision resolution (chaining, open addressing).",
        "Advanced sorting: merge sort, quick sort, radix sort, bucket sort.",
        "Complexity comparison of all sorting algorithms.",
        "",
        "TEXT BOOKS:",
        "T1. Horowitz, Sahni and Anderson-Freed, 'Fundamentals of Data Structures",
        "    in C', 2nd Edition, Universities Press, 2007.",
        "T2. Cormen, Leiserson, Rivest and Stein, 'Introduction to Algorithms',",
        "    3rd Edition, MIT Press, 2009.",
        "",
        "INTERNAL ASSESSMENT SCHEDULE:",
        "   Test 1: Units 1 and 2 -- Week 8 of Semester",
        "   Test 2: Units 3 and 4 -- Week 14 of Semester",
        "   Assignment: Algorithm implementation report -- Week 12",
        "",
        "COURSE OUTCOMES -- After completing CS301, students will be able to:",
        "CO1: Analyze time and space complexity using Big-O notation",
        "CO2: Implement linked list, stack, and queue data structures in C",
        "CO3: Apply tree and graph data structures to solve programming problems",
        "CO4: Select appropriate sorting/searching algorithms for given problems",
        "CO5: Design hash tables and resolve collisions using standard techniques",
    ])

    add_page(doc, "CS301 Syllabus -- Page 3", [
        "LABORATORY: CS301L -- Data Structures Lab",
        "Credits: 2  |  Hours/Week: 3 hours  |  Marks: 40 (ESE) + 60 (IA)",
        "",
        "List of Experiments:",
        "1.  Implementation of linear search and binary search",
        "2.  Sorting: bubble sort, insertion sort, selection sort",
        "3.  Singly linked list operations (insert, delete, reverse)",
        "4.  Stack using array and linked list",
        "5.  Infix to postfix conversion using stack",
        "6.  Queue and circular queue implementation",
        "7.  Binary Search Tree: insert, delete, search, traversals",
        "8.  Graph traversal: BFS and DFS",
        "9.  Dijkstra's shortest path algorithm",
        "10. Hash table with collision resolution by chaining",
        "11. Merge sort and quick sort with timing comparison",
        "12. Mini project: Student record management using BST",
        "",
        "MAPPING TO PROGRAM OUTCOMES:",
        "CO1 -> PO1, PO2",
        "CO2 -> PO1, PO5",
        "CO3 -> PO1, PO2, PO5",
        "CO4 -> PO1, PO2",
        "CO5 -> PO1, PO5",
    ])

    save(doc, "syllabus_cs301.pdf")


# ── Document 5: Placement Policy ─────────────────────────────────────────────

def create_placement_policy():
    doc = new_pdf()

    add_page(doc, "PLACEMENT POLICY -- Greenfield College of Engineering (2024-25)", [
        "1. ELIGIBILITY CRITERIA FOR CAMPUS PLACEMENTS",
        "",
        "To be eligible to participate in on-campus placement drives, a",
        "student must satisfy ALL of the following conditions:",
        "",
        "(a) Academic performance:",
        "    - Minimum CGPA of 6.0 up to and including Semester 6",
        "    - No active backlog (failed subject) at the time of placement",
        "    - Maximum of 2 cleared arrears (not pending) is allowed",
        "",
        "(b) Attendance:",
        "    - Minimum overall attendance of 75% in current semester",
        "",
        "(c) Registration:",
        "    - Registered with the Training and Placement (T&P) Cell",
        "      by 1 August 2024",
        "    - Updated resume uploaded to the placement portal",
        "",
        "(d) Fee clearance:",
        "    - All college fee dues must be cleared before appearing for",
        "      any company's selection process",
        "",
        "2. ONE COMPANY OFFER POLICY",
        "",
        "A student who has received and accepted a confirmed job offer",
        "(letter of intent or offer letter) is NOT eligible to participate",
        "in further placement drives during the same academic year.",
        "",
        "Dream Company Drives (packages above Rs. 12 LPA) are exempt",
        "from this rule -- all registered students may apply.",
    ])

    add_page(doc, "PLACEMENT POLICY -- Page 2", [
        "3. PLACEMENT DRIVE CONDUCT",
        "",
        "Students must carry the following documents to every drive:",
        "   - College ID card (mandatory)",
        "   - 3 copies of updated resume",
        "   - 2 passport-size photographs",
        "   - Original and photocopies of all mark sheets",
        "   - Original and photocopy of 10th and 12th certificates",
        "",
        "Students must report 30 minutes before the scheduled start time.",
        "Mobile phones must be switched off during all selection rounds.",
        "Use of mobile phones during aptitude tests or interviews will",
        "result in disqualification and a 30-day ban from all placement",
        "activities.",
        "",
        "4. PACKAGE CATEGORIES",
        "",
        "Companies are categorized by the annual package (CTC) offered:",
        "",
        "   Category A  (Mass Recruiter): Up to Rs. 5 LPA",
        "   Category B  (Mid-level):      Rs. 5 LPA to Rs. 12 LPA",
        "   Category C  (Dream Company):  Above Rs. 12 LPA",
        "",
        "Students who receive a Category A offer may still participate in",
        "Category B and C drives. Once a Category B offer is accepted,",
        "the student may participate only in Category C drives.",
        "",
        "5. INTERNSHIP CONVERSION (PPO)",
        "",
        "Pre-placement offers (PPO) arising from Semester 6 internships",
        "are treated as placement offers. Students holding a PPO must",
        "declare it to the T&P Cell by 15 July 2024.",
    ])

    add_page(doc, "PLACEMENT POLICY -- Page 3", [
        "6. OFF-CAMPUS PLACEMENT SUPPORT",
        "",
        "The T&P Cell provides support for off-campus placements:",
        "   - Resume review and feedback sessions",
        "   - Mock aptitude tests every Saturday (10 AM to 12 PM)",
        "   - Technical interview preparation workshops (monthly)",
        "   - HR interview simulation sessions (by appointment)",
        "",
        "Students securing off-campus offers must inform the T&P Cell",
        "within 7 days of receiving the offer letter.",
        "",
        "7. PLACEMENT STATISTICS TARGET (2024-25)",
        "",
        "The T&P Cell aims for the following outcomes this academic year:",
        "   - 85% placement rate for eligible registered students",
        "   - Median package: Rs. 6.5 LPA",
        "   - Highest package target: Rs. 30 LPA",
        "   - Number of companies visiting campus: minimum 50",
        "",
        "Note: These are targets, not guarantees.",
        "",
        "8. CONTACT",
        "",
        "Training and Placement Cell, Greenfield College of Engineering",
        "Room 105, Administrative Block",
        "Email: placement@greenfield.edu.in",
        "Phone: 044-2234-5678 (Monday to Friday, 9 AM to 5 PM)",
        "T&P Officer: Mr. Suresh Babu",
    ])

    save(doc, "placement_policy.pdf")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Creating sample PDFs in data/docs/ ...")
    create_attendance_policy()
    create_exam_rules()
    create_academic_regulations()
    create_syllabus()
    create_placement_policy()
    print()
    print("Done! 5 sample PDFs created.")
    print("Replace these with your real PDFs when ready.")
