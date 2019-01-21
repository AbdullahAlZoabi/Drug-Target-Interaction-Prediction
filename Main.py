import pandas as pd
import numpy as np
import DataReadWrite
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import statistics
import math
import HubnessAware
import SimpleWeightedProfile
import JaccardSimilarity



#======================================================================================  Kinase

#OriginalData = DataReadWrite.ReadOriginalKinase();

#DDOriginalSimilarity = OriginalData["DDSimilarity"];

#TTOriginalSimilarity = OriginalData["TTSimilarity"];

#Interactions = OriginalData["Interactions"];

#JaccardSimilarity.Run(DDOriginalSimilarity,TTOriginalSimilarity,Interactions,5,1)



#======================================================================================= Nuclear Receptors (NR)


InteractionsNR,DDSimilarityNR,TTSimilarityNR = DataReadWrite.load_data_from_file("nr", 'datasets')


HubnessAware.Run(DDSimilarityNR,TTSimilarityNR,InteractionsNR,5)  

    





            
        

    




