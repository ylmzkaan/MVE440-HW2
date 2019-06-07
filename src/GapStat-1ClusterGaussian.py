import numpy as np
from DataGeneration import generateData
from gapStat import optimalK
from matplotlib import pyplot as plt

# featureMeanAndStd = np.array([[0,5],[1,1]]) # [[mean1,mean2,...],[std1,std2,...]]
featureMeanAndStd = np.array([[0,5],[10,10]]) # [[mean1,mean2,...],[std1,std2,...]]

distribution = "normal"
nDataPointsToPlot = np.linspace(20,400, num=4, dtype=int)

plt.subplots(2,4)
plt.tight_layout()
for i,nDataPoints in enumerate(nDataPointsToPlot):
    data = generateData(featureMeanAndStd, nDataPoints, distribution=distribution)

    k, gapdf = optimalK(data)

    plt.subplot(2,4,2*i+1)
    dataToPlot = data.transpose()
    plt.scatter(dataToPlot[0], dataToPlot[1])
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("{} Data Points - Gap Statistic found {} cluster(s)".format(nDataPoints, k))

    plt.subplot(2,4,2*i+2)
    plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
    plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
    plt.grid(True)
    plt.xlabel('Cluster Count')
    plt.ylabel('Gap Value')

plt.show()
