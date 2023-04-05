import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Data prakiraan cuaca (suhu, kelembaban, tekanan udara, kecepatan angin)
X = np.array([[28, 60, 1012, 5], [25, 50, 1015, 10], [26, 70, 1008, 7], [22, 80, 1005, 8], [30, 65, 1010, 4],
              [20, 55, 1018, 6], [24, 75, 1013, 9], [27, 65, 1011, 3], [23, 70, 1015, 10], [21, 60, 1020, 5]])
y = np.array(['Cerah', 'Berawan', 'Hujan', 'Hujan', 'Cerah', 'Hujan', 'Hujan', 'Cerah', 'Berawan', 'Hujan'])

# Normalisasi data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Pelatihan Model
k = 3
classifier = KNeighborsClassifier(n_neighbors=k)
classifier.fit(X, y)

# Prediksi cuaca pada data baru
input_data = []
detail = ['Suhu : ', 'Kelembapan : ', 'Tekanan Udara : ', 'Kecepatan Angin : ']
for i in range(4):
    input_data.append(int(input(detail[i])))

new_data = np.array(input_data, dtype=int).reshape(1, -1)
new_data = scaler.transform(new_data)
predictions = classifier.predict(new_data)

# Cetak hasil prediksi
print(f"Prakiraan cuaca pada data baru : {predictions[0]}")
