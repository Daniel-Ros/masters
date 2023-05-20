from matplotlib.backend_bases import colors
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib

class Wrapper:
    def __init__(self):
        self.times = 1
        self.d = 2 
        self.num_of_gausians = 3

    def run(self):
        for _run in range(self.times):
            for d in range(100,101):
                for t in range(d):
                    phi = np.random.normal(0,1,(d,d))
                    X,Y = self.gen_data(1000,self.gen_means(self.num_of_gausians,self.d),self.gen_covs(self.num_of_gausians,self.d))
                    print(len(X))
                    print(len(X[0]))
                    self.run_one_time(t,d,phi,X,Y)
                    break


    def run_one_time(self, t,d, phi,X,Y):
        plt.scatter(X[:,0],X[:,1],c=Y)
        plt.axis('equal')
        plt.show()

    def gen_data(self,size,means,cov):
        dataset = []
        labels  = []
        for i in range(size):
            y = np.random.randint(len(means))
            x = np.random.default_rng().multivariate_normal(means[y],cov[y])
        
            dataset.append(x)
            labels.append(y)

        return np.array(dataset),np.array(labels)

    def gen_means(self,size,d):
        dataset = []
        for i in range(size):
            mean = np.random.normal(0,5,d)
            dataset.append(mean)
        return dataset
    
    def gen_covs(self,size,d):
        dataset = []
        for i in range(size):
            cov = np.identity(d)
            dataset.append(cov)
        return dataset


