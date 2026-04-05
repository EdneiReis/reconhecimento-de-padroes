import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Carregar a base de dados Iris
iris = load_iris()
X = iris.data
y = iris.target

# Configurações solicitadas na atividade
tamanhos_treino = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 10% a 80%
num_rodadas = 30 # Execuções iterativas para gerar as métricas de variação
resultados = []

# Classificador NN (Nearest Neighbor equivale ao K-NN com K=1)
clf_nn = KNeighborsClassifier(n_neighbors=1)

print("Processando...")

# 2. Laço para variar o tamanho da tabela inicial (treino)
for tamanho in tamanhos_treino:
    acertos_rodada = []
    
    for _ in range(num_rodadas):
        # Separação aleatória mantendo a proporção das classes (stratify)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=tamanho, stratify=y
        )
        
        # Treinamento e Teste
        clf_nn.fit(X_train, y_train)
        y_pred = clf_nn.predict(X_test)
        
        # Coleta da taxa de acerto
        acerto = accuracy_score(y_test, y_pred)
        acertos_rodada.append(acerto)
    
    # 3. Síntese dos resultados para a tabela
    acertos_rodada = np.array(acertos_rodada)
    resultados.append({
        "Treino (%)": int(tamanho * 100),
        "Taxa Média": np.mean(acertos_rodada),
        "Taxa Máxima": np.max(acertos_rodada),
        "Taxa Mínima": np.min(acertos_rodada),
        "Variância": np.var(acertos_rodada)
    })

# 4. Apresentar Tabela
df_resultados = pd.DataFrame(resultados)
print("\n" + "="*65)
print("SÍNTESE DOS RESULTADOS - CLASSIFICADOR NN (BASE IRIS)")
print("="*65)
print(df_resultados.to_string(index=False, float_format="%.4f"))
print("="*65 + "\n")

# 5. Apresentar Gráfico da Taxa Média de Acertos
plt.figure(figsize=(8, 5))
plt.plot(df_resultados["Treino (%)"], df_resultados["Taxa Média"], marker='o', linestyle='-', color='blue')
plt.title("Classificador NN: Taxa Média de Acertos vs Tamanho do Treino")
plt.xlabel("Tamanho da Tabela Inicial (Treino) em %")
plt.ylabel("Taxa Média de Acerto")
plt.xticks(df_resultados["Treino (%)"])
plt.grid(True, linestyle='--', alpha=0.7)

# Salva a imagem no diretório e exibe na tela
plt.savefig("grafico_ex1_nn.png", dpi=300, bbox_inches='tight')
print("Gráfico salvo como 'grafico_ex1_nn.png'. Feche a janela do gráfico para encerrar o script.")
plt.show()