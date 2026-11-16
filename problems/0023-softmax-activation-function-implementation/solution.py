import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    sc = []
    for i in scores:
        x = i-np.max(scores)
        sc.append(x)

    listr = []
    for i in sc:
        r = np.exp(i)/sum(np.exp(sc))
        listr.append(r)
    return listr