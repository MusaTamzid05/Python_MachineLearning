from matplotlib.colors import ListedColormap;
import numpy as np
from matplotlib import pyplot as plt

class DecisionTree_Plotter:

    def __init__(self , X , y , classifier , resolution = 0.2 ):

        self.X = X
        self.y = y
        self.classifier = classifier
        self.resolution = resolution

        self.markers  = ( "s" , "x" , "o" , "^"  , "v")
        self.colors = ( "red" , "blue" , "lightgreen" , "gray" , "cyan" )
        self.cmap = ListedColormap(self.colors[:len(np.unique(y))])

        self.init_mesh_grid_()


    def init_mesh_grid_(self):


        self.x1_min , self.x1_max = self.X[: , 0].min() - 1 , self.X[: , 0].max() + 1
        self.x2_min , self.x2_max = self.X[: , 1].min() - 1 , self.X[: , 1].max() + 1

        self.xx1 , self.xx2 = np.meshgrid(np.arange(self.x1_min , self.x2_max , self.resolution),
                np.arange(self.x2_min , self.x2_max , self.resolution))


        self.Z = self.classifier.predict(np.array([self.xx1.ravel() , self.xx2.ravel()]).T)
        self.Z = self.Z.reshape(self.xx1.shape)
        print("Grid created.")


    def draw(self):

        plt.contourf(self.xx1 , self.xx2 , self.Z , alpha = 0.4 ,
                cmap = self.cmap)
        plt.xlim(self.xx1.min() , self.xx1.max())
        plt.xlim(self.xx2.min() , self.xx2.max())


        for idx , cl in enumerate(np.unique(self.y)):
            plt.scatter(x = self.X[ self.y == cl , 0 ] , y = self.X[ self.y == cl , 1 ] , alpha = 0.8 , c = self.cmap(idx) , label = cl)


