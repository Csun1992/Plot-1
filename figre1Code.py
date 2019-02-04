from lxml.html import parse
from sys import exit
from urllib2 import urlopen
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




microsoft = \
"https://finance.yahoo.com/quote/MSFT/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo" 
apple = \
"https://finance.yahoo.com/quote/AAPL/history?period1=628495200&period2=1535691600&interval=1mo&filter=history&frequency=1mo"
att = \
"https://finance.yahoo.com/quote/T/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
ford = \
"https://finance.yahoo.com/quote/F/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo" 
sony = \
"https://finance.yahoo.com/quote/SNE/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
gap = \
"https://finance.yahoo.com/quote/GPS/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
fedex = \
"https://finance.yahoo.com/quote/FDX/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
mcdonalds = \
"https://finance.yahoo.com/quote/MCD/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
nike = \
"https://finance.yahoo.com/quote/NKE/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
tiffany = \
"https://finance.yahoo.com/quote/TIF/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
homeDepot = \
"https://finance.yahoo.com/quote/HD/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
walmart = \
"https://finance.yahoo.com/quote/WMT/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
cocaCola = \
"https://finance.yahoo.com/quote/KO/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
avon = \
"https://finance.yahoo.com/quote/AVP/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
oracle = \
"https://finance.yahoo.com/quote/ORCL/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
ibm = \
"https://finance.yahoo.com/quote/IBM/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
intel = \
"https://finance.yahoo.com/quote/INTC/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
harleyDavidson = \
"https://finance.yahoo.com/quote/HOG/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
toyota = \
"https://finance.yahoo.com/quote/TM/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
honda = \
"https://finance.yahoo.com/quote/HMC/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
boeing = \
"https://finance.yahoo.com/quote/BA/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
jpmorgan = \
"https://finance.yahoo.com/quote/JPM/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
boa = \
"https://finance.yahoo.com/quote/BAC/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
amgen = \
"https://finance.yahoo.com/quote/AMGN/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
hermanMiller = \
"https://finance.yahoo.com/quote/MLHR/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
nissan = \
"https://finance.yahoo.com/quote/NSANY/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
generalElectric = \
"https://finance.yahoo.com/quote/GE/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
nextEra = \
"https://finance.yahoo.com/quote/NEE/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
conocoPhillips = \
"https://finance.yahoo.com/quote/COP/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
bakerHughes = \
"https://finance.yahoo.com/quote/BHGE/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
dukeEnergy = \
"https://finance.yahoo.com/quote/DUK/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"
chevron = \
"https://finance.yahoo.com/quote/CVX/history?period1=625903200&period2=1549173600&interval=1mo&filter=history&frequency=1mo"


if __name__ == '__main__':
    companyName = ['microsoft', 'apple', 'att', 'ford', 'sony', 'gap', 'fedex', 'mcdonalds', 'nike',
    'tiffany', 'homeDepot', 'walmart', 'cocaCola', 'avon', 'oracle', 'ibm', 'intel',
    'harley-davidson', 'toyota', 'honda', 'boeing', 'jpmorgan', 'boa', 'amgen', 'hermanMiller',
    'nissan', 'generalElectric', 'nextEra', 'conocoPhillips', 'bakerHughes', 'dukeEnergy', 'chevron']
    
    websites = [microsoft, apple, att, ford, sony, gap, fedex, mcdonalds, nike,
    tiffany, homeDepot, walmart, cocaCola, avon, oracle, ibm, intel,
    harleyDavidson, toyota, honda, boeing, jpmorgan, boa, amgen, hermanMiller,
    nissan, generalElectric, nextEra, conocoPhillips, bakerHughes, dukeEnergy, chevron]
    
#    for company in range(len(websites)):
#        parsed = parse(urlopen(websites[company]))
#        doc = parsed.getroot()
#        tables = doc.findall('.//table')
#        table = tables[0] 
#        contents = table.findall('.//tr')
#        data = [unpack(contents[0], kind='th')] 
#        for row in contents[1:-1]:
#            data.append(unpack(row))
#        toDateTime = lambda x : datetime.strptime(x, "%b %d, %Y")
#        index = [toDateTime(data[i][0]) for i in range(1, len(data))]
#        data = pd.DataFrame(data[1:len(data)], columns=data[0], index = index)
#        data = data.dropna()[['Adj Close**', 'Volume']]
#        data.columns = ['AdjClose', 'Volume']
#        getRidComma = lambda x : x.replace(',', '')
#        data.Volume = data.Volume.map(getRidComma)
#        data = data.astype('float32', copy=False)
#        print data
#        exit()
#        dataFileName = 'data/' + companyName[company] + '.txt'
#        np.savetxt(dataFileName, data)
    



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


    priceInfo = getInputData('apple')
    aggData = np.concatenate((priceInfo, clusterData), axis=1)
    columnNames = ['current price', 'past month', '2 month ago',\
        'volume', 'unemploy.', 'inflation', 'DJIA', 'S&P']
    aggData = pd.DataFrame(aggData, columns=columnNames)
    from pandas.plotting import scatter_matrix
    scatter_matrix(aggData, figsize=(12,12))
    plt.show()
