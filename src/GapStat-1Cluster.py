import numpy as np
from DataGeneration import generateData
from gapStat import optimalK
from matplotlib import pyplot as plt

featureRanges = np.array([[0,5],[5,10]]) # [[low1,low2,...],[high1,high2,...]]
distribution = "uniform"
nDataPointsToPlot = np.linspace(20,1000, num=4, dtype=int)

plt.subplots(2,4)
plt.tight_layout()
for i,nDataPoints in enumerate(nDataPointsToPlot):
    data = generateData(featureRanges, nDataPoints, distribution=distribution)

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
