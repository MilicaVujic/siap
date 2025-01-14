import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


af_data = pd.read_csv('afrika.csv')
ir_data = pd.read_csv('irak.csv')
pk_data = pd.read_csv('pakistan.csv')

required_columns = ['pol', 'godina_studija', 'oblast', 'drzava', 'ocena', 
                    'sati_ucenja_nedeljno', 'prisustvo_na_nastavi', 'smestaj', 
                    'finansijski_status', 'bliskost_sa_roditeljima']

empty_dataset = pd.DataFrame(columns=required_columns)

def calculate_combined_boundaries(africa_grades, iraq_avg_grades):
    all_grades = np.concatenate([africa_grades, iraq_avg_grades])
    all_grades = all_grades[~np.isnan(all_grades)]

    mean = np.mean(all_grades)
    std_dev = np.std(all_grades)

    lower_bound_outliers = mean - 2 * std_dev
    upper_bound_outliers = np.percentile(all_grades, 99)  

    filtered_grades = all_grades[(all_grades >= lower_bound_outliers) & (all_grades <= upper_bound_outliers)]

    kmeans = KMeans(n_clusters=3, random_state=42).fit(filtered_grades.reshape(-1, 1))
    centers = sorted(kmeans.cluster_centers_.flatten())

    upper_boundary = (centers[1] + centers[2]) / 2 + 3  # Postavljanje gornje granice između drugog i trećeg centra
    boundaries = [min(filtered_grades), (centers[0] + centers[1]) / 2 + 3, upper_boundary, max(filtered_grades)]
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_grades, bins=20, color='gray', alpha=0.7, label="Distribucija ocena")
    
    plt.axvline(lower_bound_outliers, color='red', linestyle='--', label='Donja granica outliera')
    plt.axvline(upper_bound_outliers, color='red', linestyle='--', label='Gornja granica outliera')
    plt.axvline(boundaries[1], color='green', linestyle='--', label=f'Granica 1: {boundaries[1]:.2f}')
    plt.axvline(boundaries[2], color='green', linestyle='--', label=f'Granica 2: {boundaries[2]:.2f}')

    plt.title("Distribucija ocena studenata iz Afrike i Iraka")
    plt.xlabel("Score")
    plt.ylabel("Broj studenata")
    plt.legend(loc='upper left')

    plt.show()

    return [boundaries[1], boundaries[2]]


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


def transform_african_data(df, boundaries):
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

    transformed['ocena'] = process_grades_and_map(df, 'Your Matric (grade 12) Average/ GPA (in %)', boundaries)

    transformed['sati_ucenja_nedeljno'] = df['Additional amount of studying (in hrs) per week'].apply(
        lambda x: float(str(x).strip().replace('+', '')) if pd.notna(x) and str(x).strip().replace('+', '').isdigit() else '')

    transformed['prisustvo_na_nastavi'] = df['How many classes do you miss per week due to alcohol reasons, (i.e: being hungover or too tired?)'].apply(
        lambda x: 'vgood' if x in [0, 1] else ('good' if x in [2, 3] else 'poor'))

    transformed['smestaj'] = df['Your Accommodation Status Last Year (2023)'].apply(
        lambda x: 'Private' if 'Private' in str(x) else ('Non private' if 'Non-private' in str(x) else ''))

    transformed['finansijski_status'] = 'good'

    transformed['bliskost_sa_roditeljima'] = df['How strong is your relationship with your parent/s?']

    return transformed

def transform_iraqi_data(df, boundaries):
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
    transformed['ocena'] = process_grades_and_map(df, 'Avg', boundaries)

    transformed['sati_ucenja_nedeljno'] = df['Study Hour'].apply(lambda x: float(x) * 7 if pd.notna(x) else '')

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

boundaries = calculate_combined_boundaries(africa_grades, iraq_avg_grades)

transformed_af_data = transform_african_data(af_data, boundaries)
transformed_ir_data = transform_iraqi_data(ir_data, boundaries)
transformed_pk_data = transform_pakistani_data(pk_data)

empty_dataset = pd.concat([transformed_af_data, transformed_ir_data, transformed_pk_data], ignore_index=True)

empty_dataset = empty_dataset.dropna(subset=['ocena','pol','godina_studija','oblast','smestaj'])

empty_dataset['sati_ucenja_nedeljno'] = empty_dataset['sati_ucenja_nedeljno'].replace('', np.nan)
empty_dataset['sati_ucenja_nedeljno'] = pd.to_numeric(empty_dataset['sati_ucenja_nedeljno'], errors='coerce')
empty_dataset['sati_ucenja_nedeljno'] = empty_dataset['sati_ucenja_nedeljno'].interpolate(method='polynomial', order=3)
empty_dataset['sati_ucenja_nedeljno'] = empty_dataset['sati_ucenja_nedeljno'].round(1)


empty_dataset.to_csv('objedinjeni.csv', index=False)

print("Podaci su uspešno transformisani i sačuvani u fajlu objedinjeni.csv.")


#modeli
#rf
data = empty_dataset

X = data.drop(columns=['ocena', 'pol'])  
y = data['ocena']  

label_encoder = LabelEncoder()

for column in X.select_dtypes(include=['object']).columns:
    X[column] = label_encoder.fit_transform(X[column])

y = label_encoder.fit_transform(y)


X['prisustvo_na_nastavi'] = X['prisustvo_na_nastavi'].replace('', np.nan)
X['prisustvo_na_nastavi'] = pd.to_numeric(X['prisustvo_na_nastavi'], errors='coerce')
X['prisustvo_na_nastavi'] = X['prisustvo_na_nastavi'].interpolate(method='polynomial', order=3)
X['prisustvo_na_nastavi'] = X['prisustvo_na_nastavi'].round(0)


X['finansijski_status'] = X['finansijski_status'].replace('', np.nan)
X['finansijski_status'] = pd.to_numeric(X['finansijski_status'], errors='coerce')
X['finansijski_status'] = X['finansijski_status'].interpolate(method='polynomial', order=3)
X['finansijski_status'] = X['finansijski_status'].round(0)


X['bliskost_sa_roditeljima'] = X['bliskost_sa_roditeljima'].replace('', np.nan)
X['bliskost_sa_roditeljima'] = pd.to_numeric(X['bliskost_sa_roditeljima'], errors='coerce')
X['bliskost_sa_roditeljima'] = X['bliskost_sa_roditeljima'].interpolate(method='polynomial', order=3)
X['bliskost_sa_roditeljima'] = X['bliskost_sa_roditeljima'].round(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
X_valid=scaler.transform(X_valid)


# KNN Model
knn_model = KNeighborsClassifier()
knn_param_grid = {
    'n_neighbors': [13,15],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'kd_tree'],
    'metric': ['euclidean', 'manhattan'],
    'leaf_size': [5, 10],
    'p': [1, 2]
}

knn_grid_search = GridSearchCV(estimator=knn_model, param_grid=knn_param_grid, cv=7, n_jobs=-1, verbose=10)
knn_grid_search.fit(X_train, y_train)

best_knn_model = knn_grid_search.best_estimator_
y_pred_knn = best_knn_model.predict(X_valid)
knn_accuracy = accuracy_score(y_valid, y_pred_knn)
print(f"Tačnost k-NN modela: {knn_accuracy:.4f}")

y_pred_knn_test = best_knn_model.predict(X_test)
knn_test_accuracy = accuracy_score(y_test, y_pred_knn_test)
print(f"Tačnost k-NN modela na test skupu: {knn_test_accuracy:.4f}")

#RF
rf = RandomForestClassifier(
    n_estimators=100,  
    max_depth=None,    
    random_state=42,
    class_weight='balanced'    
)

rf.fit(X_train, y_train)

y_test_pred = rf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))