from scipy.stats import rv_continuous
import numpy as np

class pdistribution(rv_continuous):
    def _pdf(self,x):
        return 0.5 * (1 + x)

def computef(X):
    return 3/2 * X**2 * (1 + X)

def main():
    pobject = pdistribution(a = -1., b = 1.)
    for i in [10, 1000, 10000]:
        X = pobject.rvs(size = (i,))
        F = computef(X)
        print ("Samples = " + str(i))
        print ("Mean = " + str(np.mean(F)))
        print ("Variance = " + str(np.var(F)))
        print (" ")




if __name__ == "__main__":
    main()
