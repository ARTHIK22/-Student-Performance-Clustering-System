import pandas as pd
import numpy as np
import os

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "Riley", "Avery", "Dakota",
    "Emma", "Liam", "Noah", "Olivia", "Ava", "Ethan", "Sophia", "Mason", "Isabella", "Jacob",
    "Mia", "William", "Harper", "Evelyn", "James", "Benjamin", "Lucas", "Charlotte", "Amelia", "Logan",
    "Aarav", "Ananya", "Diya", "Rohan", "Priya", "Kabir", "Vihaan", "Aditi", "Sai", "Ishaan"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Sharma", "Patel", "Verma", "Rao", "Gupta", "Kumar", "Singh", "Reddy", "Nair", "Joshi"
]

def generate_student_dataset(n_samples=300, random_seed=42):
    """
    Generates a realistic student performance dataset with natural clustering patterns.
    Clusters represent:
    1. High Performers (High Study Hours, High Attendance, High Marks)
    2. At-Risk / Struggling (Low Study Hours, Low Attendance, Low Marks)
    3. Passive Learners (Low/Moderate Study Hours, High Attendance, Moderate Marks)
    4. Independent Learners (High Study Hours, Low/Moderate Attendance, Good Marks)
    """
    np.random.seed(random_seed)
    
    # Proportions of clusters
    n1 = int(n_samples * 0.30)  # High Performers
    n2 = int(n_samples * 0.25)  # At-Risk
    n3 = int(n_samples * 0.25)  # Passive Learners
    n4 = n_samples - (n1 + n2 + n3)  # Independent Learners
    
    # 1. High Performers
    study_1 = np.random.normal(loc=22, scale=4, size=n1)
    attend_1 = np.random.normal(loc=88, scale=5, size=n1)
    assign_1 = np.random.normal(loc=85, scale=6, size=n1)
    exam_1 = np.random.normal(loc=86, scale=6, size=n1)
    
    # 2. At-Risk
    study_2 = np.random.normal(loc=6, scale=2.5, size=n2)
    attend_2 = np.random.normal(loc=52, scale=8, size=n2)
    assign_2 = np.random.normal(loc=50, scale=8, size=n2)
    exam_2 = np.random.normal(loc=48, scale=9, size=n2)
    
    # 3. Passive Learners (High attendance, low study hours)
    study_3 = np.random.normal(loc=8, scale=2.5, size=n3)
    attend_3 = np.random.normal(loc=85, scale=6, size=n3)
    assign_3 = np.random.normal(loc=68, scale=7, size=n3)
    exam_3 = np.random.normal(loc=65, scale=7, size=n3)
    
    # 4. Independent Learners (High study hours, lower attendance)
    study_4 = np.random.normal(loc=20, scale=3.5, size=n4)
    attend_4 = np.random.normal(loc=60, scale=7, size=n4)
    assign_4 = np.random.normal(loc=76, scale=7, size=n4)
    exam_4 = np.random.normal(loc=78, scale=7, size=n4)
    
    # Combine data
    study_hours = np.concatenate([study_1, study_2, study_3, study_4])
    attendance = np.concatenate([attend_1, attend_2, attend_3, attend_4])
    assignment = np.concatenate([assign_1, assign_2, assign_3, assign_4])
    exam = np.concatenate([exam_1, exam_2, exam_3, exam_4])
    
    # Clip to realistic ranges
    study_hours = np.clip(np.round(study_hours, 1), 1.0, 35.0)
    attendance = np.clip(np.round(attendance, 1), 35.0, 100.0)
    assignment = np.clip(np.round(assignment, 1), 30.0, 100.0)
    exam = np.clip(np.round(exam, 1), 25.0, 100.0)
    
    # Generate names and IDs
    student_ids = [f"STU-{1001 + i}" for i in range(n_samples)]
    firsts = np.random.choice(FIRST_NAMES, size=n_samples)
    lasts = np.random.choice(LAST_NAMES, size=n_samples)
    student_names = [f"{f} {l}" for f, l in zip(firsts, lasts)]
    
    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Student_Name": student_names,
        "Study_Hours_Per_Week": study_hours,
        "Attendance_Percentage": attendance,
        "Assignment_Score": assignment,
        "Exam_Score": exam
    })
    
    # Shuffle dataframe
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return df

def save_sample_dataset():
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", "student_performance_sample.csv")
    df = generate_student_dataset(n_samples=300, random_seed=42)
    df.to_csv(filepath, index=False)
    return filepath, df

if __name__ == "__main__":
    path, df = save_sample_dataset()
    print(f"Sample dataset saved to {path} with {len(df)} records.")
