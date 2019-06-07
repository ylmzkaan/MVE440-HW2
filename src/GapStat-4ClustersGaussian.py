import numpy as np
from DataGeneration import generateData
from gapStat import optimalK
from matplotlib import pyplot as plt
"""
featureMeanAndStd1 = np.array([[0,0],[1,1]]) # [[mean1,mean2,...],[std1,std2,...]]
featureMeanAndStd2 = np.array([[10,10],[1,1]]) # [[mean1,mean2,...],[std1,std2,...]]
featureMeanAndStd3 = np.array([[15,15],[1,1]]) # [[mean1,mean2,...],[std1,std2,...]]
featureMeanAndStd4 = np.array([[20,20],[1,1]]) # [[mean1,mean2,...],[std1,std2,...]]
"""

featureMeanAndStd1 = np.array([[0,0],[3,3]]) # [[mean1,mean2,...],[std1,std2,...]]
featureMeanAndStd2 = np.array([[10,10],[3,3]]) # [[mean1,mean2,...],[std1,std2,...]]
featureMeanAndStd3 = np.array([[15,15],[3,3]]) # [[mean1,mean2,...],[std1,std2,...]]
featureMeanAndStd4 = np.array([[20,20],[3,3]]) # [[mean1,mean2,...],[std1,std2,...]]


distribution = "normal"
nDataPointsToPlot = np.linspace(5, 50, num=4, dtype=int)

plt.subplots(2,4)
plt.tight_layout()
for i,nDataPoints in enumerate(nDataPointsToPlot):
    cluster1 = generateData(featureMeanAndStd1, nDataPoints, distribution=distribution)
    cluster2 = generateData(featureMeanAndStd2, nDataPoints, distribution=distribution)
    cluster3 = generateData(featureMeanAndStd3, nDataPoints, distribution=distribution)
    cluster4 = generateData(featureMeanAndStd4, nDataPoints, distribution=distribution)

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
