import numpy as np
from DataGeneration import generateData
from gapStat import optimalK
from matplotlib import pyplot as plt

featureRanges1 = np.array([[0,5],[5,10]]) # [[low1,low2,...],[high1,high2,...]]
featureRanges2 = np.array([[5,10],[10,15]]) # [[low1,low2,...],[high1,high2,...]]
featureRanges3 = np.array([[10,15],[15,20]]) # [[low1,low2,...],[high1,high2,...]]
featureRanges4 = np.array([[15,20],[20,25]]) # [[low1,low2,...],[high1,high2,...]]

distribution = "uniform"
nDataPointsToPlot = np.linspace(5, 60, num=4, dtype=int)

plt.subplots(2,4)
plt.tight_layout()
for i,nDataPoints in enumerate(nDataPointsToPlot):
    cluster1 = generateData(featureRanges1, nDataPoints, distribution=distribution)
    cluster2 = generateData(featureRanges2, nDataPoints, distribution=distribution)
    cluster3 = generateData(featureRanges3, nDataPoints, distribution=distribution)
    cluster4 = generateData(featureRanges4, nDataPoints, distribution=distribution)

    data = np.concatenate((cluster1, cluster2, cluster3, cluster4), axis=0)
    np.random.shuffle(data)

    k, gapdf = optimalK(data)

    plt.subplot(2,4,2*i+1)
    dataToPlot = data.transpose()
    plt.scatter(dataToPlot[0], dataToPlot[1])
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("{} Data Points - Gap Statistic found {} cluster(s)".format(4*nDataPoints, k))

    plt.subplot(2,4,2*i+2)
    plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
    plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
    plt.grid(True)
    plt.xlabel('Cluster Count')
    plt.ylabel('Gap Value')

plt.show()
