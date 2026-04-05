import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Configurações de bases e parâmetros
datasets = {
    "Iris": load_iris(),
    "Wine": load_wine()
}
valores_k = [1, 3, 5]
tamanhos_treino = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
num_rodadas = 30

# 2. Loop principal por Base de Dados e por valor de K
for nome_base, data in datasets.items():
    X, y = data.data, data.target
    
    for k in valores_k:
        print(f"Processando Base: {nome_base} | K={k}...")
        resultados = []
        clf = KNeighborsClassifier(n_neighbors=k)

        for tamanho in tamanhos_treino:
            acertos_rodada = []
            
            for _ in range(num_rodadas):
                # Stratify garante que todas as classes estejam no treino/teste
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=tamanho, stratify=y, random_state=None
                )
                
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acertos_rodada.append(accuracy_score(y_test, y_pred))
            
            acertos_rodada = np.array(acertos_rodada)
            resultados.append({
                "Treino (%)": int(tamanho * 100),
                "Média": np.mean(acertos_rodada),
                "Máximo": np.max(acertos_rodada),
                "Mínimo": np.min(acertos_rodada),
                "Variância": np.var(acertos_rodada)
            })

        # 3. Gerar Tabela
        df_res = pd.DataFrame(resultados)
        print(f"\n--- SÍNTESE: {nome_base} (K={k}) ---")
        print(df_res.to_string(index=False, float_format="%.4f"))
        
        # Salvar tabela em TXT para facilitar seu relatório
        df_res.to_csv(f"resultado_k{k}_{nome_base}.txt", sep='\t', index=False)

        # 4. Gerar e Salvar Gráfico
        plt.figure(figsize=(8, 5))
        plt.plot(df_res["Treino (%)"], df_res["Média"], marker='o', label=f'K={k}')
        plt.title(f"K-NN ({nome_base}): Taxa Média vs Treino (K={k})")
        plt.xlabel("Treino (%)")
        plt.ylabel("Taxa Média de Acerto")
        plt.grid(True)
        plt.savefig(f"grafico_k{k}_{nome_base}.png")
        plt.close() # Fecha para não sobrecarregar a memória

