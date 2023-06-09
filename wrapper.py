from matplotlib.backend_bases import colors
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
from sklearn import mixture
from itertools import permutations
from stats import Stats
import time
import json

from utils import calc_bat

class Wrapper:
    def __init__(self):
        self.times = 1
        self.d = 32
        self.num_of_gausians = 2
        self.num_of_samples = 50000
        self.stats = Stats()
        self.sparse = 10
        self.alinged = self.sparse
        self.noise = 0.000001

    def run(self):
        for s in range(1,33):
            self.sparse = s

            self.means = self.gen_means(self.num_of_gausians,self.d)
            self.covs = self.gen_covs(self.num_of_gausians,self.d)
            X_full,Y_full = self.gen_data(self.num_of_samples,self.means,self.covs)

            d = self.d

            for _run in range(self.times):
                phi = np.random.normal(0,1,(self.d,self.d))

                X = np.zeros_like(X_full)
                X[:,:d] = X_full[:,:d]
                Y = Y_full

                clf = mixture.GaussianMixture(n_components=self.num_of_gausians, covariance_type="full")
                clf.fit(X)

                # origin space
                clf.means_ = np.array(self.means)
                clf.covariances_ = np.array(self.covs)
                clf.weights_ = np.array([0.5,0.5])
                Z = clf.predict(X)
                Z = self.permute(Y,Z,self.num_of_gausians)
                base_err = (Y != Z).sum()
            
                for t in range(2,d):    
                    self.run_one_time(t,d,phi[:d,:t],X,Y,base_err)
        self.stats.store_resualts(f"res_sparse_alinged_cords_with_bach.csv")

    def run_one_time(self, t,d, phi_o,X,Y,base_err):
        U, S ,VH = np.linalg.svd(phi_o,full_matrices=False)
        phi = U 

        means = self.means @ phi
        covs = phi.T @ self.covs @ phi

        batch =  calc_bat(0.5,0.5,phi,self.means[0], self.means[1] , self.covs[0] , self.covs[1],self.noise,t)

        #  target space
        Xt = X @ phi
        Xt[:] += np.random.default_rng().multivariate_normal(np.zeros(t) ,np.identity(t) * self.noise)
        clft = mixture.GaussianMixture(n_components=self.num_of_gausians, covariance_type="full")
        clft.fit(Xt)
        clft.means_ = np.array(means)
        clft.covariances_ = np.array(covs)
        clft.weights_ = np.array([0.5,0.5])
        Zt = clft.predict(Xt)
        Zt = self.permute(Y,Zt,self.num_of_gausians)
        new_err = (Y != Zt).sum()

        self.stats.add_resualt(t,d,base_err / self.num_of_samples ,new_err / self.num_of_samples, batch , self.sparse)
        print(t,d,base_err / self.num_of_samples ,new_err / self.num_of_samples , batch , self.sparse)

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
        means = []
        for i in range(size):
            mean = np.random.normal(0,1,d)
            means.append(mean)
        v1 = (1/np.sqrt(d)) * np.ones(d)
        slice = np.arange(self.sparse,d)
        v1[slice] = 0
        v2 = -v1
        return [v1,v2]
    
    def gen_covs(self,size,d):
        covs = []
        for i in range(size):
            sig = 4
            cov = sig * np.identity(d)
            if self.sparse != -1:
                slice = np.arange(self.sparse,d)
                for ix in slice:
                    cov[ix,ix] = 0
                    pass
            covs.append(cov)
        return covs

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
    
    def dump(self, X,Y , means, cov):
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        obj = {
            "x" : X.tolist(),
            "y" : Y.tolist(),
            "means": means,
            "cov" : cov,
        }

        json_object = json.dumps(obj, indent=4,cls=NumpyEncoder)
        with open(f"dump_{time.time()}.json", "w") as outfile:
            outfile.write(json_object)
