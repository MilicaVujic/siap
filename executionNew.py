import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import re


af_data = pd.read_csv('afrikaNovi.csv')
ir_data = pd.read_csv('irak.csv')
pk_data = pd.read_csv('pakistan.csv')
pr_data = pd.read_csv('Portuguese.csv')

#kolona ponavlja oznacava da li ponavlja predmet ili godinu
required_columns = ['pol', 'godina_studija', 'oblast', 'drzava', 'ocena', 
                    'sati_ucenja_nedeljno', 'prisustvo_na_nastavi', 'smestaj', 
                    'finansijski_status', 'bliskost_sa_roditeljima', 'u_romanticnoj_vezi', 'ponavlja']

empty_dataset = pd.DataFrame(columns=required_columns)

def categorize_attendance(data, column):
    attendance_values = data[column].dropna()

    values = attendance_values.values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(values)
    centers = sorted(kmeans.cluster_centers_.flatten())
    boundaries = [
        min(values), 
        (centers[0] + centers[1]) / 2, 
        (centers[1] + centers[2]) / 2, 
        max(values)
    ]

    plt.figure(figsize=(8, 5))
    plt.hist(attendance_values, bins=15, color='skyblue', alpha=0.7, label='Raspodela prisustva')
    for boundary in boundaries[1:3]:  
        plt.axvline(boundary, color='red', linestyle='--', label=f'Granična vrednost: {boundary:.2f}')
    plt.title('Distribucija vrednosti prisustva studenata iz Pakistana sa izracunatim granicama')
    plt.xlabel('Prisustvo ( u procentima)')
    plt.ylabel('Broj studenata')
    plt.legend()
    plt.show()
    return boundaries[1:3]



def process_grades_and_map(data, column, boundaries):
    grades = data[column].astype(float).values

    def grade_mapping(score):
        if score <= boundaries[0]:
            return 'C'
        elif boundaries[0] < score <= boundaries[1]:
            return 'B'
        return 'A'

    return np.vectorize(grade_mapping)(grades)

def process_grades_and_map_africa(grade):
    if grade < 71:
        return 'C'
    elif grade > 81.7:
        return 'A'
    else:
        return 'B'

def process_grades_and_map_iraq(grade):
    if grade < 51:
        return 'C'
    elif grade > 75:
        return 'A'
    else:
        return 'B'

def process_grades_and_map_portugal(grade):
    if grade < 10:
        return 'C'
    elif grade > 15:
        return 'A'
    else:
        return 'B'

def classify_afr_finance(range_str):
    range_str = str(range_str)
    match = re.search(r'(\d{4})', range_str)
    
    if match:
        first_number = int(match.group(1))
        if first_number == 4001:
            return 'poor'
        elif first_number == 5001:
            return 'good'
        elif first_number > 6000:
            return 'vgood'
        else:
            return 'poor'
    else:
        return 'poor'

def classify_african_presence(absence):
    absence = str(absence)
    if absence == '0' or absence == '1':
        return 'vgood'
    elif absence == '2' or absence == '3':
        return 'good'
    else:
        return 'poor'
    
def classify_african_repeaters(num_of_fails):
    num_of_fails = str(num_of_fails)
    if num_of_fails == '0' or num_of_fails == '':
        return 'no'
    else:
        return 'yes' 

def classify_portugal_presence(absence):
    if absence > 9:
        return 'poor'
    elif absence < 4:
        return 'vgood'
    else:
        return 'poor'

def transform_african_data(df):
    transformed = pd.DataFrame(columns=required_columns)

    transformed['pol'] = df['Your Sex?'].apply(lambda x: 'F' if str(x).strip().lower() == 'female' else 'M')

    year_mapping = {
        '1st Year': 1,
        '2nd Year': 2,
        '3rd Year': 3,
        '4th Year': 4
    }
    transformed['godina_studija'] = df['What year were you in last year (2023) ?'].map(year_mapping)

    transformed['oblast'] = df['What faculty does your degree fall under?']

    transformed['drzava'] = 'Africa'

    transformed['ocena'] = df['Your Matric (grade 12) Average/ GPA (in %)'].apply(process_grades_and_map_africa)

    transformed['sati_ucenja_nedeljno'] = df['Additional amount of studying (in hrs) per week'].apply(
        lambda x: float(str(x).strip().replace('+', '')) if pd.notna(x) and str(x).strip().replace('+', '').isdigit() else '')

    transformed['prisustvo_na_nastavi'] = df['How many classes do you miss per week due to alcohol reasons, (i.e: being hungover or too tired?)'].apply(classify_african_presence)

    transformed['smestaj'] = df['Your Accommodation Status Last Year (2023)'].apply(
        lambda x: 'Private' if 'Private' in str(x) else ('Non private' if 'Non-private' in str(x) else 'Private')) ## racunam mozda umesto '' ubaci 'Private'

    transformed['finansijski_status'] = df['Monthly Allowance in 2023'].apply(classify_afr_finance)

    transformed['bliskost_sa_roditeljima'] = df['How strong is your relationship with your parent/s?']

    transformed['u_romanticnoj_vezi'] = df['Are you currently in a romantic relationship?']

    transformed['ponavlja'] = df['How many modules have you failed thus far into your studies?'].apply(classify_african_repeaters)

    return transformed

def transform_iraqi_data(df):
    transformed = pd.DataFrame(columns=required_columns)

    transformed['pol'] = df['Sex'].apply(lambda x: 'F' if str(x).strip().lower() == 'female' else 'M')

    transformed['godina_studija'] = df['Age']

    specialization_mapping = {
        'BIO': 'Biology',
        'APP': 'Engineering'
    }
    transformed['oblast'] = df['Specialization'].map(specialization_mapping)

    transformed['drzava'] = 'Iraq'

    df['Avg'] = (df['Avg1'] + df['Avg2']) / 2
    transformed['ocena'] = df['Avg'].apply(process_grades_and_map_iraq)

    transformed['sati_ucenja_nedeljno'] = df['Study Hour']

    transformed['prisustvo_na_nastavi'] = df['Attendance']

    transformed['smestaj'] = df['Home Ownership'].apply(lambda x: 'Private' if x.strip().lower() == 'own' else 'Non private')

    transformed['finansijski_status'] = df['Family Economic Level']

    family_relationship_mapping = {
        'good': 'Close',
        'excellent': 'Very close',
        'vgood': 'Very close',
        'sobad': 'Fair',
        'bad': 'Fair'
    }
    transformed['bliskost_sa_roditeljima'] = df['Family Relationship'].map(family_relationship_mapping)

    transformed['u_romanticnoj_vezi'] = df['Social Status'].apply(
        lambda x: 'No' if x.strip().lower() == 'single' else 'Yes')

    transformed['ponavlja'] = df['Failure Year']

    return transformed

def transform_portugal_data(df):
    transformed = pd.DataFrame(columns=required_columns)

    transformed['pol'] = df['sex']

    year_mapping = {
        15: 1,
        16: 2,
        17: 3,
        18: 4
    }

    transformed['godina_studija'] = df['age'].map(year_mapping)

    transformed['oblast'] = 'Engineering'

    transformed['drzava'] = 'Portugal'

    transformed['ocena'] = df['G3'].apply(process_grades_and_map_portugal)

    transformed['sati_ucenja_nedeljno'] = df['studytime']

    transformed['prisustvo_na_nastavi'] = df['absences'].apply(classify_portugal_presence)

    transformed['smestaj'] = df['Pstatus'].apply(lambda x: 'Private' if x.strip() == 'T' else 'Non private')

    transformed['finansijski_status'] = df['internet'].apply(lambda x: 'poor' if x.strip() == 'no' else 'good')

    family_relationship_mapping = {
        5: 'Very close',
        4: 'Very close',
        3: 'Close',
        2: 'Close',
        1: 'Fair'
    }
    transformed['bliskost_sa_roditeljima'] = df['famrel'].map(family_relationship_mapping)

    transformed['u_romanticnoj_vezi'] = df['romantic'].apply(
        lambda x: 'No' if x.strip() == 'no' else 'Yes')

    transformed['ponavlja'] = df['failures'].apply(lambda x: 'no' if x == 0 else 'yes')

    return transformed


def transform_pakistani_data(df):
    transformed = pd.DataFrame(columns=required_columns)

    transformed['pol'] = df['Gender']

    def year_of_study_mapping(age):
        if pd.isna(age):
            return None
        try:
            age = int(age)
            if age in [18, 19]:
                return 1
            elif age == 20:
                return 2
            elif age == 21:
                return 3
            elif age == 22:
                return 4
            else:
                return None
        except ValueError:
            return None

    transformed['godina_studija'] = df['Age'].apply(year_of_study_mapping)

    transformed['oblast'] = df['Major']
    transformed['drzava'] = 'Pakistan'
    transformed['ocena'] = df['Grades']

    transformed['sati_ucenja_nedeljno'] = df['Study_Hours'].apply(lambda x: float(x) if pd.notna(x) else '')
    boundaries_attendance=categorize_attendance(pk_data,"Attendance")

    def attendance_quality(value):
        if pd.isna(value):
            return ''
        try:
            value = float(value)
            if value > boundaries_attendance[1]:
                return 'vgood'
            elif boundaries_attendance[0] <= value <= boundaries_attendance[1]:
                return 'good'
            else:
                return 'poor'
        except ValueError:
            return ''

    transformed['prisustvo_na_nastavi'] = df['Attendance'].apply(attendance_quality)

    transformed['smestaj'] = df['Study_Space'].apply(
        lambda x: 'Private' if str(x).strip().lower() == 'yes' else ('Non private' if str(x).strip().lower() == 'no' else ''))

    financial_mapping = {
        'Low': 'poor',
        'Medium': 'good',
        'High': 'vgood'
    }
    transformed['finansijski_status'] = df['Financial_Status'].apply(
        lambda x: financial_mapping.get(str(x).strip(), '') if pd.notna(x) else ''
    )

    parental_mapping = {
        'Low': 'Fair',
        'Medium': 'Close',
        'High': 'Very close'
    }
    transformed['bliskost_sa_roditeljima'] = df['Parental_Involvement'].apply(
        lambda x: parental_mapping.get(str(x).strip(), '') if pd.notna(x) else ''
    )

    return transformed

af_data = af_data.dropna(subset=['Your Matric (grade 12) Average/ GPA (in %)'])
ir_data = ir_data.dropna(subset=['Avg1', 'Avg2'])

africa_grades = af_data['Your Matric (grade 12) Average/ GPA (in %)']
iraq_avg_grades = ((ir_data['Avg1'] + ir_data['Avg2']) / 2)
# ovo sam promenio malo racunanje ocena

transformed_af_data = transform_african_data(af_data)
transformed_ir_data = transform_iraqi_data(ir_data)
transformed_pk_data = transform_pakistani_data(pk_data)
transformed_pr_data = transform_portugal_data(pr_data)

empty_dataset = pd.concat([transformed_af_data, transformed_ir_data, transformed_pr_data], ignore_index=True)
empty_dataset = empty_dataset.dropna(subset=['ocena','pol','godina_studija','oblast','smestaj'])

empty_dataset.to_csv('nesto.csv', index=False)

print("Podaci su uspešno transformisani i sačuvani u fajlu Afr-Ir-Port.csv.")
