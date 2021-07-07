from sklearn.cluster import KMeans
from Enviroment.enviroment import *
from utils import *
from sklearn import preprocessing

class cluster_class:
    def __init__(self,n_clusters=5,n_features=8):
        self.n_clusters = n_clusters
        self.n_features = n_features
        
        self.means = np.zeros((self.n_clusters,self.n_features))
        self.stds = np.zeros((self.n_clusters,self.n_features)) + 1
           
    def train(self,start,end):
        
        env = Enviroment(start = start,end = end,variance = 0)
        
        array = np.zeros((env.num_stocks,total_timesteps,self.n_features))
        for stock in env.stocks:
            stock_index = env.stocks.index(stock)
            for day in range(start,end):
                features = env.GOAT_features(stock,timestep = day)
                
                array[stock_index,day,:] = features
                
        data_for_clustering = array.reshape((array.shape[0]*array.shape[1],array.shape[2]))
        boolean_table = data_for_clustering != np.zeros((1,array.shape[2]))
        boolean_table = np.all(boolean_table,axis = 1)
        data_for_clustering = data_for_clustering[boolean_table]
        
        self.scaler = preprocessing.StandardScaler().fit(data_for_clustering)
        data_for_clustering = self.scaler.transform(data_for_clustering)
        
        init = 'random' if (self.means == np.zeros((self.n_clusters,self.n_features))).all() else self.cluster.cluster_centers_
        n_init = 10 if (self.means == np.zeros((self.n_clusters,self.n_features))).all() else 1
        
        clusters = KMeans(n_clusters = self.n_clusters,init = init, n_init = n_init, random_state = 464354).fit(data_for_clustering)     
        self.means = self.scaler.inverse_transform(clusters.cluster_centers_)
        
        stds = []
        for i in range(self.n_clusters):
            std = np.std(data_for_clustering[clusters.labels_ == i],axis = 0)
            
            stds.append(std)
            
        self.stds = np.array(stds)
        self.cluster = clusters
    def find_features(self,array):
        
        z0 = array.shape[0]
        z1 = array.shape[1]
        z2 = array.shape[2]
        
        #print(array[1,150,2])       
        array = array.reshape((z0*z1,z2))       
        #print(array[150+2871,2])    
        array = self.scaler.transform(array)      
        #print(array[150+2871,2])    
        array = array.reshape(z0,z1,z2)      
        #print(array[1,150,2])
        
        means = self.cluster.cluster_centers_
        
        final_features = np.concatenate([(((array - means[i,:])**2))/(self.stds[i,:]) for i in range(self.n_clusters)],axis = 2)
        
        final_features = np.exp(-final_features)
        
        return final_features
        

if __name__ == '__main__':
    # inertia = []
    # for n_clusters in range(1,21):
        # c = cluster_class(n_clusters= n_clusters)
        # end = 400
        # c.train(50,end)
        
        # inertia.append(c.cluster.inertia_)
        
    # import matplotlib.pyplot as plt
    # inertia = [i/10000 for i in inertia]
    # plt.plot(range(1,21),inertia)
    # plt.title('Number of Clusters vs Inertia in warm up period')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Inertia (in 10K)')
    # plt.show()
    
    list_ = []
    
    c = cluster_class(n_clusters = 5)
    end = 400
    
    c.train(50,end)
    list_.append(c.means[1,:])
    
    # while end <= 1700:
        # end += 50
        
        # c.train(end-200,end)
        # list_.append(c.means[1,:])
    
    # list_ = np.array(list_)