import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('results/results.csv')
a1 = data[data.fase == "teste"]['akurasi'].values
p1 = data[data.fase == "teste"]['imagem'].values
# a2 = data[data.fase == "treino"]['akurasi'].values
# p2 = data[data.fase == "treino"]['imagem'].values
plt.plot(p1, a1, '-')
# plt.plot(p2, a2, '--')
plt.legend(['Eficácia de Treino'], loc='lower right')
plt.xlabel("Número da imagem")
plt.ylabel("Eficácia %")
plt.show()
