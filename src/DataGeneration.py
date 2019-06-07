import numpy as np

def generateData(featureData, nDataPoints, distribution="uniform"):

    nFeatures = len(featureData[0])
    if distribution.lower() == "uniform":
        data = np.random.uniform(low=featureData[0], high=featureData[1],
                                size=(nDataPoints,nFeatures))
    elif distribution.lower() == "normal" or distribution.lower() == "gaussian":
        data = np.random.normal(loc=featureData[0], scale=featureData[1],
                                size=(nDataPoints,nFeatures))
    else:
        raise Exception("Invalid distribution!")

    return data
