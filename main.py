import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import pandas as pd

def comp_R(A):
    n, m = A.shape
    fl = 1
    R = A.copy()
    RP = A.copy()

    while fl == 1:
        for i in range(m):
            for j in range(m):
                R[i, j] = np.max(np.min([R[i, :], A[:, j]], axis=0))

        for i in range(m):
            for j in range(m):
                R[i, j] = max(R[i, j], RP[i, j])

        if np.sum(np.sum(RP - R)) == 0:
            fl = 0

        RP = R.copy()

    return R

def comp(MAT, n):

    I = np.eye(MAT.shape[0])
    RP = I.copy()
    R = np.zeros_like(RP)

    for ep in range(n):
        A = MAT[:, :, ep]
        for i in range(A.shape[1]):
            for j in range(A.shape[1]):
                R[i, j] = np.max(np.min([RP[i, :], A[:, j]], axis=0))

        for i in range(A.shape[1]):
            for j in range(A.shape[1]):
                R[i, j] = max(R[i, j], RP[i, j])

        RP = R.copy()

    RC = []
    for i in range(A.shape[1]):
        for j in range(i + 1, A.shape[1]):
            RC.append(RP[i, j])

    D = (1 - np.array(RC)) / np.linalg.norm(1 - np.array(RC))

    return R, D


def L_sim(x, y, p):
    m = x.shape[0]
    expr1 = 1 - x ** p + y ** p
    expr2 = 1 - y ** p + x ** p

    # Verifica se ci sono valori negativi o complessi nelle espressioni
    mask1 = expr1 < 0
    mask2 = expr2 < 0

    # Gestisci i valori negativi o complessi in modo appropriato
    expr1 = np.where(mask1, 0, expr1)
    expr2 = np.where(mask2, 0, expr2)

    # Calcola le radici p-esime solo per i valori non negativi o complessi
    expr1 = np.where(mask1, 1, np.power(expr1, 1 / p))
    expr2 = np.where(mask2, 1, np.power(expr2, 1 / p))

    # Calcola i minimi
    a = np.min([expr1, np.ones(m)], axis=0)
    b = np.min([expr2, np.ones(m)], axis=0)

    u = 1/m * np.sum(np.minimum(a, b))

    return u


def integration_script(x, isFuzzy):
    # Number of temporal datasets
    k = len(x)
    # Norm
    p = 1
    # Number of observations for each channel
    n = 150

    # Example of data set
    #for i in range(k):
        #x[i]['data'] = np.random.rand(1024, channel)

    Cd = np.zeros((n, n, k))

    for ib in range(k):
        for mod_1 in range(n):
            for mod_2 in range(mod_1 + 1, n):
                Cd[mod_1, mod_2, ib] = L_sim(x[ib]['data'][mod_1, :], x[ib]['data'][mod_2, :],p) if isFuzzy else np.sqrt(np.sum((x[ib]['data'][mod_1, :] - x[ib]['data'][mod_2, :]) ** 2))
                Cd[mod_2, mod_1, ib] = Cd[mod_1, mod_2, ib]

        # Normalizza ogni colonna separatamente
        col_norms = np.linalg.norm(Cd[:, :, ib], axis=0)
        Cd[:, :, ib] /= col_norms[np.newaxis, :]
        Cd[:, :, ib] /= col_norms[:, np.newaxis]

        Cd[:, :, ib] = comp_R(Cd[:, :, ib])

        print(f'Matrix = {ib}')

    """for ib in range(k):

        for mod_1 in range(n):
            for mod_2 in range(mod_1 + 1, n):
                Cd[mod_1, mod_2, ib] = L_sim(x[ib]['data'][mod_1, :], x[ib]['data'][mod_2, :],
                                             p) if isFuzzy else np.sqrt(
                    np.sum((x[ib]['data'][mod_1, :] - x[ib]['data'][mod_2, :]) ** 2))
                Cd[mod_2, mod_1, ib] = Cd[mod_1, mod_2, ib]

        Cd[:, :, ib] = Cd[:, :, ib] / np.linalg.norm(Cd[:, :, ib])
        Cd[:, :, ib] = comp_R(Cd[:, :, ib])

        print(f'Matrix = {ib}')"""

    n = Cd.shape[2]
    R, D = comp(Cd, n)
    D = np.real(D)


    ZC = linkage(D, method='ward')
    C = fcluster(ZC, t=3, criterion='maxclust')
    print("Etichette dei Cluster:")
    unique_elements, counts = np.unique(C, return_counts=True)
    for cluster_label, count in zip(unique_elements, counts):
        print(f"Cluster {cluster_label}: {count} elementi")
    # Ora puoi utilizzare la variabile C per ulteriori analisi o visualizzazioni

    plt.figure(1)
    H = dendrogram(ZC, p=0)
    plt.title('Dendrogramma H')
    plt.show()

    # You can use the clusters in variable C for further analysis

def createDataSet(tumor):
    omicsData = ["exp", "mirna", "methy"]
    dataSets = [{} for _ in range(3)]
    i = 0
    for omic in omicsData:
        _ = np.loadtxt(tumor+"/"+omic+".txt", dtype=str, usecols=range(173))
        _ = np.delete(_, 0, axis=0)  # Rimuovere la prima riga
        _ = np.delete(_, 0, axis=1)  # Rimuovere la prima colonna
        _ = _.astype(float)
        dataSets[i]['data'] = _
        i += 1

    return dataSets

def createDataSetIris():
    file_csv = 'iris.csv'
    df = pd.read_csv(file_csv)
    dataSets = [{} for _ in range(2)]

    dataSets[0]['data'] = df[['sepal.length', 'sepal.width']].values
    dataSets[1]['data'] = df[['petal.length', 'petal.width']].values
    print(dataSets)
    return dataSets


isFuzzy = True
breast = createDataSet("breast")
sepal = createDataSetIris()
#integration_script(breast, isFuzzy)

integration_script(sepal, isFuzzy)

