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

#modeli
#rf
data = pd.read_csv('Afr-Ir-Port.csv')

X = data.drop(columns=['ocena', 'pol', 'u_romanticnoj_vezi'])  
y = data['ocena']  

label_encoder = LabelEncoder()

for column in X.select_dtypes(include=['object']).columns:
    X[column] = label_encoder.fit_transform(X[column])

y = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
X_valid=scaler.transform(X_valid)


# KNN Model

knn_model = KNeighborsClassifier(
    algorithm = 'auto',
    leaf_size = 5,
    metric = 'manhattan',
    n_neighbors = 15,
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

#RF
rf = RandomForestClassifier(
    n_estimators=100,  
    max_depth=10,    
    random_state=42,
    min_samples_split=15,
    min_samples_leaf=5   
)

rf.fit(X_train, y_train)

#Pokazuje koliko je koja kolona bitna za krajnji rezultat
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
for idx in indices:
    print(f"Feature {idx}: {importances[idx]}")

y_test_pred = rf.predict(X_test)
print("RF Accuracy:", accuracy_score(y_test, y_test_pred))

#mlp
mlp = MLPClassifier(
    hidden_layer_sizes=(10, 20, 20, 20, 15),
    max_iter=700,               
    activation='relu',
    solver='adam',    
    random_state=42,
    learning_rate_init = 0.001,
    alpha = 0.001
)

mlp.fit(X_train, y_train)

y_test_pred = mlp.predict(X_test)

print("MLP Accuracy:", accuracy_score(y_test, y_test_pred))

X = [[3,            #godina
      'Arts',       #oblast
      'Portugal',   #drzava
      3,            #sati ucenja
      'vgood',      #prisustvo
      'Private',    #smestaj
      'vgood',      #finansije
      'Very close', #odnos sa roditeljima
      'no']]        #ponavlja (godinu, predmet)

for i in range(len(X)):
    X[i] = label_encoder.fit_transform(X[i])

X = scaler.transform(X)
predicted_output = mlp.predict(X)

#problem je sto ne znam sta je [0], [1], [2]
print("Predicted Output:", predicted_output)