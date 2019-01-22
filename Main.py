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


# Uncomment one dataset pertime 
#======================================================================================  Kinase

#OriginalData = DataReadWrite.ReadOriginalKinase();

#DDOriginalSimilarity = OriginalData["DDSimilarity"];

#TTOriginalSimilarity = OriginalData["TTSimilarity"];

#Interactions = OriginalData["Interactions"];


#SimpleWeightedProfile.Run(DDOriginalSimilarity,TTOriginalSimilarity,Interactions,6);


#HubnessAware.Run(DDOriginalSimilarity,TTOriginalSimilarity,Interactions,6);



#======================================================================================= Nuclear Receptors (NR)


#InteractionsNR,DDSimilarityNR,TTSimilarityNR = DataReadWrite.load_data_from_file("nr", 'datasets')


#SimpleWeightedProfile.Run(DDSimilarityNR,TTSimilarityNR,InteractionsNR,6)


#HubnessAware.Run(DDSimilarityNR,TTSimilarityNR,InteractionsNR,6)






#======================================================================================= GPCR


#Interactions ,DDSimilarity ,TTSimilarity  = DataReadWrite.load_data_from_file("gpcr", 'datasets')


#HubnessAware.Run(DDSimilarity ,TTSimilarity ,Interactions ,6)



#======================================================================================= Ion Channels


#Interactions ,DDSimilarity ,TTSimilarity  = DataReadWrite.load_data_from_file("ic", 'datasets')


#HubnessAware.Run(DDSimilarity ,TTSimilarity ,Interactions ,6)


#======================================================================================= Enzyme


Interactions ,DDSimilarity ,TTSimilarity  = DataReadWrite.load_data_from_file("e", 'datasets')


print("6")
SimpleWeightedProfile.Run(DDSimilarity ,TTSimilarity ,Interactions ,6)





            
        

    




