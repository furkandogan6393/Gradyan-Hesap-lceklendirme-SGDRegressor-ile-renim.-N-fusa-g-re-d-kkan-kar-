import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


def dataset():
    x = np.array([6.1101, 5.5277, 8.5186, 7.0032, 5.8598]).reshape(-1, 1)
    y = np.array([17.592, 9.1302, 13.662, 11.854, 6.8233])
    return x,y

x_train, y_train=dataset()

scaler = StandardScaler()
x_train_scaled= scaler.fit_transform(x_train) #Ölçekleme yaptık.

#Dönüş sayısı vb. değerleri atıyoruz.
model = SGDRegressor(max_iter=15000,tol=1e-3,alpha=0.01,random_state=42)

#ölçeklendirilmiş x_train ve y_train verimizi eğittik.
model.fit(x_train_scaled,y_train)

#Değerlerimizi ekrana veriyoruz.
print(f"Model katsayısı w: {model.coef_[0]}")
print(f"Model katsayısı b: {model.intercept_[0]}")

#kullanıcıdan girdi
nüfus_input = float(input("Nüfusu Seçiniz: "))
nüfus = np.array([[nüfus_input]])
nüfus_scaled = scaler.transform(nüfus)   
y_pred=model.predict(nüfus_scaled)


#Performans 
y_train_pred=model.predict(x_train_scaled)
mse = mean_squared_error(y_train, y_train_pred)
r2 = r2_score(y_train, y_train_pred)
print(f"Ortalama Kare Hatası (MSE): {mse:.2f}")
print(f"R Kare Skoru (R²): {r2:.2f}")



print(f"{nüfus_input:.1f} milyonluk nüfus için kâr tahmini: {y_pred[0]:.2f}")

plt.scatter(x_train, y_train, color='red', label="Gerçek Veriler")
plt.plot(x_train, y_train_pred, color='blue', label="Tahmin Edilen")
plt.xlabel("Nüfus (Milyon)")
plt.ylabel("Kâr")
plt.title("SGDRegressor ile Doğrusal Regresyon")
plt.legend()
plt.show()