import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def unpack(row, kind='td'):
    line = row.findall('.//%s' % kind)
    return [val.text_content() for val in line]

def getStockPrice(fileName):
    price = []
    size = []
    with open(fileName) as f:
        for line in f:
            item = line.rstrip().split(',')
            price.append(float(item[-2]))
            size.append(float(item[-1]))
    return price, size[2:-1]

def getInputData(fileName):        
    fileName = 'data/' + fileName
    price, size = getStockPrice(fileName)
    currentPrice = price[2:-1]
    lastMonthPrice = price[1:-2]
    twoMonthPrice = price[:-3]
    inputData = np.array([currentPrice, lastMonthPrice, twoMonthPrice, size]).T
    return inputData


if __name__ == '__main__':
    unemployment = []
    with open("data/unemployment", 'r') as f:
       for line in f:
           unemployment.append(map(float, line.rstrip().rstrip('\n').split(' ')))
    unemployment = [item for sublist in unemployment for item in sublist]
    
    inflation = []
    with open("data/inflation", 'r') as f:
        for line in f:
            inflation.append(line.rstrip().rstrip('\n').split('   '))
    inflation = map(float, [item.rstrip('%') for sublist in inflation for item in sublist])
    
    djia = []
    with open("data/djia", 'r') as f:
        for line in f:
            djia.append(float(line.rstrip()))
    djia = djia[1:]
    
    sp = []
    with open('data/sp', 'r') as f:
        for line in f:
            item = line.rstrip().split(',')
            sp.append(float(item[-2]))
    sp = sp[1:]
    
    clusterData = np.array([unemployment, inflation, djia, sp]).T
    np.savetxt('data/clusterData.txt', clusterData)
   
    sampleSize = np.size(clusterData, axis=0)
    clusterNum = 3
    kmeans = KMeans(n_clusters=clusterNum, random_state=41).fit(clusterData)
    labels = kmeans.labels_

    dateIndex = pd.date_range('1/1/1990', '9/1/2018', freq='M')
    index = np.array([str(i) + '/' + str(j) for i,j in zip(dateIndex.year, dateIndex.month)])
    featureNames = ['unemployment', 'inflation', 'Dow Jones', 'S & P']
    fig, axes = plt.subplots(4, 1, sharex=True)
    xticks = np.linspace(0, sampleSize - 1, 4, dtype=np.int32)
    colors = ['r', 'g', 'k']
    clusters = ['cluster 1', 'cluster 2', 'cluster 3']
    x = np.array(range(sampleSize))
    
    for i in range(4):
        for j in range(3):
            axes[i].scatter(x[labels==j], clusterData[labels==j, i], c=colors[j], \
                    label=clusters[j])
            axes[i].set_ylabel(featureNames[i])
            axes[i].set_xticks(xticks)
            axes[i].set_xticklabels(index[xticks])
    plt.xlabel('time')
    plt.legend(loc='best')
    fig.suptitle("Clustering results based on macroeconomic features")
    plt.savefig('clusterResult.png')
