from scipy.stats import rv_continuous,norm
import numpy as np

np.random.seed(0)


class pdistribution(rv_continuous):
    def _pdf(self, x):
        return (0.5 * (1 + x))


class qdistribution(rv_continuous):
    def _pdf(self, X):
        return  ((15/16) * (X ** 2) * ((1 + X) ** 2))


def computef(X):
    return ((3/2) * (X**2) * (1 + X))


def computer(X, mean = None):
    mask = np.logical_and((X > -1), (X < 1))
    p_x = (0.5 * (1 + X)) * mask

    if mean == 0:
        q_x = norm.pdf(X, loc=0, scale=1)
    elif mean == 3:
        q_x = norm.pdf(X, loc=3, scale=1)
    else:
        q_x = ((15/16) * (X ** 2) * ((1 + X) ** 2))# * mask + 1e-5

    return p_x / q_x


def main():
    ## Q 4.2
    print ("Q 4.2\n")
    pobject = pdistribution(a = -1., b = 1., seed = 0)
    for i in [10, 1000, 10000]:
        X = pobject.rvs(size = (i,))
        F = computef(X)
        print ("Samples = " + str(i))
        print ("Mean = " + str(np.mean(F)))
        print ("Variance = " + str(np.var(F)))
        print (" ")

    ## Q 4.3
    ## N(3,1)
    print ("Q 4.3\n")
    print ("N(3,1)")
    FW3 = []
    for i in [10, 1000, 10000]:
        X = np.random.normal(loc = 3., size = (i,))
        F = computef(X)
        R = computer(X, mean=3)
        if np.sum(R) == 0:
            RW = R / (np.sum(R)+1e-3)
        else:
            RW = R / (np.sum(R))

        FW3.append(F * RW)
        print ("Samples = " + str(i))
        print ("Mean = " + str(np.mean(F * R)))
        print ("Variance = " + str(np.var(F * R)))
        print (" ")

    # N(0,1)
    print ("N(0,1)")
    FW0 = []
    for i in [10, 1000, 10000]:
        X = np.random.normal(size = (i,))
        F = computef(X)
        R = computer(X, mean=0)
        if np.sum(R) == 0:
            RW = R / (np.sum(R)+1e-3)
        else:
            RW = R / (np.sum(R))
        FW0.append(F * RW)
        print ("Samples = " + str(i))
        print ("Mean = " + str(np.mean(F * R)))
        print ("Variance = " + str(np.var(F * R)))
        print (" ")

    # Q Distribution
    print ("Q Distribution")
    qobject = qdistribution(a = -1., b = 1., seed = 0)
    FWQ = []
    for i in [10, 1000, 10000]:
        X = qobject.rvs(size = (i,))
        F = computef(X)
        R = computer(X)
        if np.sum(R) == 0:
            RW = R / (np.sum(R)+1e-3)
        else:
            RW = R / (np.sum(R))
        FWQ.append(F *  RW )
        print ("Samples = " + str(i))
        print ("Mean = " + str(np.mean(F * R)))
        print ("Variance = " + str(np.var(F * R)))
        print (" ")

    #Q 4.4
    print ("Q 4.4\n")
    i = [10 , 1000, 10000]
    print ("N(3,1)")
    for iterable in range(3):
        print ("Samples = " + str(i[iterable]))
        print ("Mean = " + str(np.mean(FW3[iterable]) * i[iterable]))
        print ("Variance = " + str(np.var(FW3[iterable])))
        print ("")

    print ("N(0,1)")
    for iterable in range(3):
        print ("Samples = " + str(i[iterable]))
        print ("Mean = " + str(np.mean(FW0[iterable]) * i[iterable]))
        print ("Variance = " + str(np.var(FW0[iterable])))
        print ("")

    print ("Q Distribution")
    for iterable in range(3):
        print ("Samples = " + str(i[iterable]))
        print ("Mean = " + str(np.mean(FWQ[iterable]) * i[iterable]))
        print ("Variance = " + str(np.var(FWQ[iterable])))
        print ("")


if __name__ == "__main__":
    main()
