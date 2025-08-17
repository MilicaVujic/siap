import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

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
