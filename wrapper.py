from matplotlib.backend_bases import colors
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
from sklearn import mixture
from itertools import permutations
from stats import Stats

class Wrapper:
    def __init__(self):
        self.times = 10
        self.d = 100
        self.num_of_gausians = 2
        self.num_of_samples = 10000
        self.stats = Stats()

    def run(self):
        X_full,Y_full = self.gen_data(self.num_of_samples,self.gen_means(self.num_of_gausians,self.d),self.gen_covs(self.num_of_gausians,self.d))
        for _run in range(self.times):
            phi = np.random.normal(0,1,(self.d,self.d))
            for d in range(50,self.d):
                X = X_full[:,:d]
                Y = Y_full
                 # origin space
                clf = mixture.GaussianMixture(n_components=self.num_of_gausians, covariance_type="full")
                Z = clf.fit_predict(X)
                Z = self.permute(Y,Z,self.num_of_gausians)
                base_err = (Y != Z).sum()

                for t in range(2,np.int32(np.floor(d/2))):    
                    self.run_one_time(t,d,phi[:d,:t],X,Y,base_err)
            self.stats.store_resualts(f"res{d}_{_run}.csv")
                    
                    
    def run_one_time(self, t,d, phi,X,Y,base_err):
        #  target space
        Xt = X @ phi
        clft = mixture.GaussianMixture(n_components=self.num_of_gausians, covariance_type="full")
        Zt = clft.fit_predict(Xt)
        Zt = self.permute(Y,Zt,self.num_of_gausians)
        new_err = (Y != Zt).sum()
        self.stats.add_resualt(t,d,base_err / self.num_of_samples ,new_err / self.num_of_samples)
        print(t,d,base_err / self.num_of_samples ,new_err / self.num_of_samples)

        

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
            mean = np.random.normal(0,1,d)
            dataset.append(mean)
        return dataset
    
    def gen_covs(self,size,d):
        dataset = []
        for i in range(size):
            sig = 5
            cov = sig * np.identity(d)
            dataset.append(cov)
        return dataset

    def permute(self,Y,Z,k):
        from_values =  np.arange(0,max(Z)+1)
        min_err = np.inf
        min_per = None
        for i in list(permutations([j for j in range(k)])):
            per = np.array(i)
            sort_idx = np.argsort(from_values)
            idx = np.searchsorted(from_values,Z,sorter = sort_idx)
            Zn = per[sort_idx][idx]
            err = np.abs(Y - Zn).sum()
            if err < min_err:
                min_err = err
                min_per = Zn


        return min_per

