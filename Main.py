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

#JaccardSimilarity.Run(DDOriginalSimilarity,TTOriginalSimilarity,Interactions,1,0,"Kinase",0)



#======================================================================================= Nuclear Receptors (NR)


#InteractionsNR,DDSimilarityNR,TTSimilarityNR = DataReadWrite.load_data_from_file("nr", 'datasets')


#JaccardSimilarity.Run(DDSimilarityNR,TTSimilarityNR,InteractionsNR,5,1,"NR",0)



#======================================================================================= GPCR


Interactions ,DDSimilarity ,TTSimilarity  = DataReadWrite.load_data_from_file("gpcr", 'datasets')


HubnessAware.Run(DDSimilarity ,TTSimilarity ,Interactions ,5)





            
        

    




