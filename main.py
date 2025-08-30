import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from graphs import calculate_combined_boundaries, categorize_attendance

# graphs
af_data = pd.read_csv('datasets/afrika.csv')
ir_data = pd.read_csv('datasets/irak.csv')
pk_data = pd.read_csv('datasets/pakistan.csv')
af_data = af_data.dropna(subset=['Your Matric (grade 12) Average/ GPA (in %)'])
ir_data = ir_data.dropna(subset=['Avg1', 'Avg2'])

africa_grades = af_data['Your Matric (grade 12) Average/ GPA (in %)']
iraq_avg_grades = ((ir_data['Avg1'] + ir_data['Avg2']) / 2)

boundaries = calculate_combined_boundaries(africa_grades, iraq_avg_grades)

#models
#rf
data = pd.read_csv('datasets/Afr-Ir-Por.csv')
data = data.replace('', np.nan)

X=data.drop(columns=['ocena'])
y = data['ocena']  

feature_encoder = LabelEncoder()
for column in X.select_dtypes(include=['object']).columns:
    X[column] = feature_encoder.fit_transform(X[column])

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
X_valid=scaler.transform(X_valid)

print('Treniranje modela za predikciju ocena studenata...\n')
# KNN Model
knn_model = KNeighborsClassifier(
    algorithm = 'auto',
    leaf_size = 5,
    metric = 'minkowski',
    n_neighbors = 23,
    p = 1,
    weights = 'uniform'
)

knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_valid)
knn_accuracy = accuracy_score(y_valid, y_pred_knn)
print(f"Tačnost k-NN modela: {knn_accuracy:.4f}")

y_pred_knn_test = knn_model.predict(X_test)
knn_test_accuracy = accuracy_score(y_test, y_pred_knn_test)
print(f"Tačnost k-NN modela na test skupu: {knn_test_accuracy:.4f}")

#Random Forest
rf = RandomForestClassifier(
    n_estimators=100,  
    max_depth=5,    
    random_state=42,
    min_samples_split=5,
    min_samples_leaf=10,   
)

rf.fit(X_train, y_train)
y_test_pred = rf.predict(X_test)
print("\nRF Accuracy:", accuracy_score(y_test, y_test_pred))

print('\nPrikaz koji atributi su najvažniji za predviđanje:')
feature_names = X.columns
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
for idx in indices:
    print(f"Feature '{feature_names[idx]}': {importances[idx]}")

#mlp
mlp = MLPClassifier(
    hidden_layer_sizes=(32, 32, 32, 32),
    max_iter=3000,
    activation='relu',
    solver='adam',    
    random_state=42,
    learning_rate_init = 0.001,
    alpha = 0.0001,
    early_stopping=True,
    n_iter_no_change=15,
)

mlp.fit(X_train, y_train)
y_test_pred = mlp.predict(X_test)
print("\nMLP Accuracy:", accuracy_score(y_test, y_test_pred))

# Primer predikcije za novog studenta
student_columns = [
    'pol',
    'godina_studija',
    'oblast',
    'drzava',
    'kolicina_ucenja',
    'prisustvo_na_nastavi',
    'smestaj',
    'finansijski_status',
    'bliskost_sa_roditeljima'
]

X_new = [[
      'M',          #pol
      3,            #godina
      'Arts',       #oblast
      'Portugal',   #drzava
      2,            #sati ucenja (bolje reci kolicina ucenja na skali 1-4)
      'vgood',      #prisustvo na nastavi
      'Private',    #smestaj
      'vgood',      #finansije
      'Very close'  #odnos sa roditeljima
]]

print("\n--------------------------------------------")
print("Primer podataka jednog studenta:")
for col, val in zip(student_columns, X_new[0]):
    print(f"{col}: {val};")
    
for i in range(len(X_new)):
    X_new[i] = feature_encoder.fit_transform(X_new[i])

X_new = scaler.transform(X_new)
predicted_output = mlp.predict(X_new)

predicted_label = target_encoder.inverse_transform(predicted_output)
print("\nPredvidjena ocena(klasa) studenta na osnovu podataka studenta:", predicted_label)