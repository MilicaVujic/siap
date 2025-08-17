import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import re

af_data = pd.read_csv('datasets/afrikaNovi.csv')
ir_data = pd.read_csv('datasets/irak.csv')
pk_data = pd.read_csv('datasets/pakistan.csv')
pr_data = pd.read_csv('datasets/Portuguese.csv')

pk_data = pk_data.dropna(subset=["Study_Space", "Financial_Status", "Parental_Involvement","Attendance"])

required_columns = ['pol', 'godina_studija', 'oblast', 'drzava', 'ocena', 
                    'sati_ucenja_nedeljno', 'prisustvo_na_nastavi', 'smestaj', 
                    'finansijski_status', 'bliskost_sa_roditeljima']

empty_dataset = pd.DataFrame(columns=required_columns)


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

def classify_portugal_presence(absence):
    if absence > 9:
        return 'poor'
    elif absence < 4:
        return 'vgood'
    else:
        return 'good'

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


def plot_grade_distribution_and_boundaries(grades, title, lower_bound, upper_bound):
    plt.figure(figsize=(8, 5))
    plt.hist(grades, bins=15, color='skyblue', alpha=0.7, label='Raspodela ocena')

    plt.axvline(lower_bound, color='red', linestyle='--', label=f'Donja granica: {lower_bound:.2f}')
    plt.axvline(upper_bound, color='green', linestyle='--', label=f'Gornja granica: {upper_bound:.2f}')

    plt.title(title)
    plt.xlabel('Ocena')
    plt.ylabel('Broj studenata')
    plt.legend()
    plt.show()


def remove_outliers_and_determine_bounds(grades):
    Q1 = grades.quantile(0.41)
    Q3 = grades.quantile(0.66)
    IQR = Q3 - Q1
    lower_bound = Q1 - IQR 
    upper_bound = Q3 + IQR 
    return lower_bound, upper_bound

def categorize_grades(grades, lower_bound, upper_bound):
    def grade_mapping(score):
        if score <= lower_bound:
            return 'C'
        elif lower_bound < score <= upper_bound:
            return 'B'
        return 'A'
    return np.vectorize(grade_mapping)(grades)

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

    grades = df['Your Matric (grade 12) Average/ GPA (in %)'].astype(float)
    lower_bound, upper_bound = remove_outliers_and_determine_bounds(grades)
    plot_grade_distribution_and_boundaries(grades, 'Distribucija ocena studenata iz Afrike', lower_bound, upper_bound-2)
    transformed['ocena'] = categorize_grades(grades, lower_bound, upper_bound)

    transformed['sati_ucenja_nedeljno'] = df['Additional amount of studying (in hrs) per week'].apply(
        lambda x: float(str(x).strip().replace('+', '')) if pd.notna(x) and str(x).strip().replace('+', '').isdigit() else '')

    transformed['prisustvo_na_nastavi'] = df['How many classes do you miss per week due to alcohol reasons, (i.e: being hungover or too tired?)'].apply(classify_african_presence)

    transformed['smestaj'] = df['Your Accommodation Status Last Year (2023)'].apply(
        lambda x: 'Private' if 'Private' in str(x) else ('Non private' if 'Non-private' in str(x) else 'Private'))

    transformed['finansijski_status'] = df['Monthly Allowance in 2023'].apply(classify_afr_finance)

    transformed['bliskost_sa_roditeljima'] = df['How strong is your relationship with your parent/s?']

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
    grades = df['Avg'].astype(float)
    lower_bound, upper_bound = remove_outliers_and_determine_bounds(grades)
    plot_grade_distribution_and_boundaries(grades, 'Distribucija ocena studenata iz Iraka',lower_bound+5, upper_bound-5)

    transformed['ocena'] = categorize_grades(grades, lower_bound, upper_bound)

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

    grades = df['G3'].astype(float)
    lower_bound, upper_bound = remove_outliers_and_determine_bounds(grades)
    plot_grade_distribution_and_boundaries(grades, 'Distribucija ocena studenata iz Portugala', lower_bound-1, upper_bound+1)
    transformed['ocena'] = categorize_grades(grades, lower_bound, upper_bound)

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
    boundaries_attendance = categorize_attendance(pk_data, "Attendance")

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

transformed_af_data = transform_african_data(af_data)
transformed_ir_data = transform_iraqi_data(ir_data)
transformed_pk_data = transform_pakistani_data(pk_data)
transformed_pr_data = transform_portugal_data(pr_data)

empty_dataset = pd.concat([transformed_af_data, transformed_ir_data, transformed_pr_data, transformed_pk_data], ignore_index=True)
empty_dataset = empty_dataset.dropna(subset=['ocena','pol','godina_studija','oblast','smestaj','bliskost_sa_roditeljima'])
empty_dataset['sati_ucenja_nedeljno'] = pd.to_numeric(empty_dataset['sati_ucenja_nedeljno'], errors='coerce')
empty_dataset['sati_ucenja_nedeljno'] = (
    empty_dataset['sati_ucenja_nedeljno']
    .interpolate(method='quadratic')
    .round(1)
)
status_mapping = {'poor': 0, 'good': 1, 'vgood': 2}
empty_dataset['finansijski_status_numeric'] = empty_dataset['finansijski_status'].map(status_mapping)

empty_dataset['finansijski_status_numeric'] = empty_dataset['finansijski_status_numeric'].interpolate(method='quadratic')

reverse_mapping = {v: k for k, v in status_mapping.items()}
empty_dataset['finansijski_status'] = empty_dataset['finansijski_status_numeric'].round().astype(int).map(reverse_mapping)

empty_dataset = empty_dataset.drop(columns=['finansijski_status_numeric'])

def calculate_percentages(df, column):
    counts = df.groupby(['drzava', column]).size().unstack(fill_value=0)
    counts['Total'] = counts.sum(axis=1)
    for value in counts.columns[:-1]:  
        counts[f'{value}_percent'] = (counts[value] / counts['Total']) * 100
    return counts

ocena_vrednosti = {'A': 3, 'B': 2, 'C': 1}

ocena_counts = calculate_percentages(empty_dataset, 'ocena')
ocena_counts['Prosecna_uspesnost'] = (
    ocena_counts['A'] * ocena_vrednosti['A'] +
    ocena_counts['B'] * ocena_vrednosti['B'] +
    ocena_counts['C'] * ocena_vrednosti['C']
) / ocena_counts['Total']

najuspesnija_drzava = ocena_counts['Prosecna_uspesnost'].idxmax()
najmanje_uspesna_drzava = ocena_counts['Prosecna_uspesnost'].idxmin()

pol_counts = calculate_percentages(empty_dataset, 'pol')
prisustvo_counts = calculate_percentages(empty_dataset, 'prisustvo_na_nastavi')
smestaj_counts = calculate_percentages(empty_dataset, 'smestaj')
bliskost_counts = calculate_percentages(empty_dataset, 'bliskost_sa_roditeljima')
finansije_counts = calculate_percentages(empty_dataset, 'finansijski_status')

sati_ucenja = empty_dataset.groupby('drzava')['sati_ucenja_nedeljno'].mean()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

ocena_counts[['A_percent', 'B_percent', 'C_percent']].plot(kind='bar', stacked=True, ax=axes[0, 0])
axes[0, 0].set_title('Distribucija ocena po državama')

pol_counts[['M_percent', 'F_percent']].plot(kind='bar', stacked=True, ax=axes[0, 1])
axes[0, 1].set_title('Distribucija pola po državama')

prisustvo_counts[["poor_percent","good_percent","vgood_percent"]].plot(kind='bar',stacked=True, ax=axes[0,2])
axes[0,2].set_title('Distribucija prisustva na nastavi po drzavama')



smestaj_counts[['Private_percent', 'Non private_percent']].plot(kind='bar', stacked=True, ax=axes[1, 0])
axes[1, 0].set_title('Smeštaj po državama')

bliskost_counts[['Very close_percent', 'Close_percent', 'Fair_percent']].plot(kind='bar', stacked=True, ax=axes[1, 1])
axes[1, 1].set_title('Bliskost sa roditeljima')

finansije_counts[['poor_percent', 'good_percent', 'vgood_percent']].plot(kind='bar', stacked=True, ax=axes[1, 2])
axes[1, 2].set_title('Finansijski status po državama')

plt.tight_layout()
plt.show()

print(f'Najuspešnija država je {najuspesnija_drzava} sa prosečnom uspešnošću {ocena_counts["Prosecna_uspesnost"].max():.2f}.')
print(f'Najmanje uspešna država je {najmanje_uspesna_drzava} sa prosečnom uspešnošću {ocena_counts["Prosecna_uspesnost"].min():.2f}.')

print("\nProsečan broj sati učenja nedeljno po državama:")
print(sati_ucenja)

transformed_pk_data = transformed_pk_data.sample(n=300, random_state=42)
empty_dataset = pd.concat([transformed_af_data, transformed_ir_data, transformed_pr_data, transformed_pk_data], ignore_index=True)
empty_dataset = empty_dataset.dropna(subset=['ocena','pol','godina_studija','oblast','smestaj','prisustvo_na_nastavi','bliskost_sa_roditeljima'])
empty_dataset['sati_ucenja_nedeljno'] = pd.to_numeric(empty_dataset['sati_ucenja_nedeljno'], errors='coerce')
empty_dataset['sati_ucenja_nedeljno'] = (
    empty_dataset['sati_ucenja_nedeljno']
    .interpolate(method='quadratic')
    .round(0)
)
#
status_mapping = {'poor': 0, 'good': 1, 'vgood': 2}
empty_dataset['finansijski_status_numeric'] = empty_dataset['finansijski_status'].map(status_mapping)

empty_dataset['finansijski_status_numeric'] = empty_dataset['finansijski_status_numeric'].interpolate(method='quadratic')

reverse_mapping = {v: k for k, v in status_mapping.items()}
empty_dataset['finansijski_status'] = empty_dataset['finansijski_status_numeric'].round().astype(int).map(reverse_mapping)

empty_dataset = empty_dataset.drop(columns=['finansijski_status_numeric'])
empty_dataset.to_csv('datasets/Afr-Ir-Por-Pak.csv', index=False)

print("Podaci su uspešno transformisani i sačuvani u fajlu 'Afr-Ir-Por-Pak.csv.")