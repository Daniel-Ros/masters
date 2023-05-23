from matplotlib.backend_bases import colors
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
from sklearn import mixture
from itertools import permutations
from stats import Stats
import time
import json

class Wrapper:
    def __init__(self):
        self.times = 25
        self.d = 100
        self.num_of_gausians = 2
        self.num_of_samples = 50000
        self.stats = Stats()
        self.sparse = np.int32(self.d / 2)

    def run(self):
        self.means = self.gen_means(self.num_of_gausians,self.d)
        self.covs = self.gen_covs(self.num_of_gausians,self.d)
        X_full,Y_full = self.gen_data(self.num_of_samples,self.means,self.covs)

        d = self.d
        X = X_full[:,:d]
        Y = Y_full
        self.dump(X,Y,self.means,self.covs)
        for _run in range(self.times):
            phi = np.random.normal(0,1,(self.d,self.d))
                       
            # origin space
            clf = mixture.GaussianMixture(n_components=self.num_of_gausians, covariance_type="full")
            clf.fit(X)
            clf.means_ = np.array(self.means)
            clf.covariances_ = np.array(self.covs)
            clf.weights_ = np.array([0.5,0.5])
            Z = clf.predict(X)
            Z = self.permute(Y,Z,self.num_of_gausians)
            base_err = (Y != Z).sum()
        
            for t in range(2,d):    
                self.run_one_time(t,d,phi[:d,:t],X,Y,base_err)
            self.stats.store_resualts(f"res_sparse_{_run}.csv")
                    
                    
    def run_one_time(self, t,d, phi_o,X,Y,base_err):
        U, S ,VH = np.linalg.svd(phi_o,full_matrices=False)
        phi = U 

        means = self.means @ phi
        covs = phi.T @ self.covs @ phi

        #  target space
        Xt = X @ phi
        clft = mixture.GaussianMixture(n_components=self.num_of_gausians, covariance_type="full")
        clft.fit(Xt)
        clft.means_ = np.array(means)
        clft.covariances_ = np.array(covs)
        clft.weights_ = np.array([0.5,0.5])
        Zt = clft.predict(Xt)
        Zt = self.permute(Y,Zt,self.num_of_gausians)
        new_err = (Y != Zt).sum()
        self.stats.add_resualt(t,d,base_err / self.num_of_samples ,new_err / self.num_of_samples)
        print(t,d,base_err ,new_err)

        

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
        v2 = v1
        return [v1,v2]
    
    def gen_covs(self,size,d):
        covs = []
        for i in range(size):
            sig = 25
            cov = sig * np.identity(d)

            if self.sparse != -1:
                # for i in range(d - self.sparse):
                #     ix = np.random.randint(d)
                #     while cov[ix,ix] == 0:
                #         ix = np.random.randint(d)
                #     print(f"{ix},", end="")
                #     cov[ix,ix] = 0
                # print("")
                per = np.random.permutation(np.arange(d))
                slice = per[: (d - self.sparse)]
                print(np.sort(per[(d - self.sparse):]))
                for ix in slice:
                    # cov[ix,ix] = 0
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

