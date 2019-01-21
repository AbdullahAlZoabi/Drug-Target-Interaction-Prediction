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


print(SimpleWeightedProfile.Run(DDOriginalSimilarity,TTOriginalSimilarity,Interactions,5))





    


    

    





            
        

    




