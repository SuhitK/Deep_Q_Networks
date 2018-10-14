from scipy.stats import rv_continuous,norm
import numpy as np

class pdistribution(rv_continuous):
    def _pdf(self,x):
        return 0.5 * (1 + x)

class qdistribution(rv_continuous):
    def _pdf(self,x):
        return  15/16 * x**2 * (1 + x)**2

def computef(X):
    return 3/2 * X**2 * (1 + X)

def computer(X, mean = None):
    mask= np.logical_and((X > -1), (X < 1))
    p_x = 0.5 * (1 + X) * mask
    if mean == 0:
        q_x = norm(0,1).pdf(X)
    elif mean == 3:
        q_x = norm(3,1).pdf(X)
    else:
        q_x = 15/16 * X ** 2 * (1 + X) ** 2 * mask

    return p_x / q_x

        
    
def main():
    np.random.seed(0)
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
        R = computer(X, 3)
        RW = R /( np.sum(R) + 1e-3)
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
        R = computer(X)
        RW = R / (np.sum(R) + 1e-3)
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
        RW = R / (np.sum(R) + 1e-3)
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
