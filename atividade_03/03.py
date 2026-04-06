import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def dmc_predict(X_train, y_train, X_test):
    classes = np.unique(y_train)
    
    centroides = {}
    for c in classes:
        centroides[c] = X_train[y_train == c].mean(axis=0)
    
    y_pred = []
    for x in X_test:
        distancias = [np.linalg.norm(x - centroides[c]) for c in classes]
        y_pred.append(classes[np.argmin(distancias)])
    
    return np.array(y_pred)

def executar(nome, loader):
    print(f"\n=== {nome} ===")
    
    data = loader()
    X = data.data
    y = data.target
    
    tamanhos = np.arange(0.1, 0.9, 0.1)
    medias = []
    
    # arquivo de resultado
    arquivo = open(f"resultado_{nome}.txt", "w")
    arquivo.write("Treino (%)  Media   Maximo  Minimo  Variancia\n")
    
    for t in tamanhos:
        resultados = []
        
        for _ in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=t)
            
            y_pred = dmc_predict(X_train, y_train, X_test)
            acc = accuracy_score(y_test, y_pred)
            resultados.append(acc)
        
        media = np.mean(resultados)
        maximo = np.max(resultados)
        minimo = np.min(resultados)
        variancia = np.var(resultados)
        
        linha = f"{int(t*100)}      {media:.4f}  {maximo:.4f}  {minimo:.4f}  {variancia:.4f}\n"
        
        print(linha)
        arquivo.write(linha)
        
        medias.append(media)
    
    # gráfico
    plt.plot(tamanhos * 100, medias, marker='o')
    plt.title(f"DMC - {nome}")
    plt.xlabel("Treino (%)")
    plt.ylabel("Taxa média")
    plt.grid()
    plt.savefig(f"grafico_{nome}.png")
    plt.clf()
    
    # matriz
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    y_pred = dmc_predict(X_train, y_train, X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    arquivo.write("\nMatriz de Confusao:\n")
    arquivo.write(str(cm))
    
    arquivo.close()


# executar
executar("Iris", datasets.load_iris)
executar("Wine", datasets.load_wine)