import numpy as np
class Stats:
    def __init__(self):
        self.db = []

    def add_resualt(self,t,d,base_err,new_err,batch,sparse):
        self.db.append([t,d,base_err,new_err,batch,sparse])

    def store_resualts(self,file):
        np.savetxt(file,np.array(self.db),delimiter=",")

    def show_resualts(self):
        pass