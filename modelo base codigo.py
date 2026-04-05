import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Carregar Base de Dados (Exemplo com Iris)
data = load_iris()
X = data.data
y = data.target

# Configurações do experimento
train_sizes = np.arange(0.1, 0.9, 0.1) # 10% a 80%
num_rodadas = 30 # Número de repetições para gerar estatísticas confiáveis

# Dicionário para armazenar os resultados
resultados = []
medias_plot = []

# 2. Laço de Tamanho da Tabela Inicial
for test_size in train_sizes:
    acertos = []
    
    # 3. Laço de Rodadas para o mesmo tamanho de treino
    for _ in range(num_rodadas):
        # Separação dos dados (test_size no sklearn é o % de TESTE, então usamos 1 - train_size)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=test_size, stratify=y
        )
        
        # DEFINIÇÃO DO CLASSIFICADOR 
        # Ex 1: NN (K=1)
        # Ex 2: K-NN (Alterar K)
        # Ex 3: DMC (NearestCentroid())
        clf = KNeighborsClassifier(n_neighbors=1) 
        
        # Treino e Predição
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Calcular Taxa de Acerto
        acc = accuracy_score(y_test, y_pred)
        acertos.append(acc)
    
    # 4. Cálculo das Métricas exigidas na Tabela
    acertos = np.array(acertos)
    media = np.mean(acertos)
    maximo = np.max(acertos)
    minimo = np.min(acertos)
    variancia = np.var(acertos)
    
    medias_plot.append(media)
    
    resultados.append({
        "Tamanho Treino (%)": round(test_size * 100),
        "Média": round(media, 4),
        "Máximo": round(maximo, 4),
        "Mínimo": round(minimo, 4),
        "Variância": round(variancia, 6)
    })

# 5. Apresentar a Tabela
df_resultados = pd.DataFrame(resultados)
print("--- SÍNTESE DOS RESULTADOS ---")
print(df_resultados.to_string(index=False))

# 6. Apresentar o Gráfico da Taxa Média
plt.figure(figsize=(8, 5))
plt.plot(df_resultados["Tamanho Treino (%)"], df_resultados["Média"], marker='o', linestyle='-', color='b')
plt.title("Taxa Média de Acertos vs Tamanho do Treino")
plt.xlabel("Tamanho da Tabela Inicial (Treino) %")
plt.ylabel("Taxa Média de Acerto")
plt.grid(True)
plt.show()

# Apenas um exemplo de como extrair a matriz
# Para visualizar a matriz de confusão, podemos usar o último modelo treinado (ou qualquer outro)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()
