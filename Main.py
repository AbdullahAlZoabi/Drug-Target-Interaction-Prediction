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

OriginalData = DataReadWrite.ReadOriginalKinase();


DDOriginalSimilarity = OriginalData["DDSimilarity"];


TTOriginalSimilarity = OriginalData["TTSimilarity"];


Interactions = OriginalData["Interactions"];


#Call this function once to calculate the Jaccard Similarities and write it to csv files 


#I1,U1 = JaccardSimilarity.DDMatrixJaccardSimilarity(Interactions);

#I2,U2 = JaccardSimilarity.TTMatrixJaccardSimilarity(Interactions);

#DataReadWrite.WriteJaccardKinase(I1,U1,I2,U2);


#JaccardData = DataReadWrite.ReadJaccardKinase();

#DDJaccardSimilarity = JaccardData["DDSimilarity"];

#TTJaccardSimilarity = JaccardData["TTSimilarity"];

#JaccardInteractions = JaccardData["Interactions"];



print(JaccardSimilarity.Run(DDOriginalSimilarity,TTOriginalSimilarity,Interactions,3,0))

#print(AllTargetsWeighting(TTOriginalSimilarity,Interactions,2));



    


    

    





            
        

    




