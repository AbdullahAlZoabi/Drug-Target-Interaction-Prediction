import pandas as pd
import numpy as np
import DataReadWrite
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc

def DrugBasedPrediction(i,j,DDSimilarity,Interactions,NumOfNeighbours):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    Numerator  = 0;

    Denominator = 0;

    SortedIndices = np.argsort(-DDSimilarity.iloc[i,:]);

    Count = 0;

    for k in range(0, NumOfDrugs):

        CurrentIndex = SortedIndices.iloc[k,];

        if CurrentIndex != i:

            Numerator = Numerator + Interactions.iloc[CurrentIndex,j]*DDSimilarity.iloc[i,CurrentIndex];

            Denominator = Denominator + DDSimilarity.iloc[i,CurrentIndex];

            Count = Count + 1;

            if Count == NumOfNeighbours:

                break;

    return Numerator/Denominator;



def TargetBasedPrediction(i,j,TTSimilarity,Interactions,NumOfNeighbours):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    Numerator  = 0;

    Denominator = 0;

    SortedIndices = np.argsort(-TTSimilarity.iloc[j,:]);

    Count = 0;

    for k in range(0, NumOfTargets):

        CurrentIndex = SortedIndices.iloc[k,];

        if CurrentIndex != j:

            Numerator = Numerator + Interactions.iloc[i,CurrentIndex]*TTSimilarity.iloc[j,CurrentIndex];

            Denominator = Denominator + TTSimilarity.iloc[j,CurrentIndex];

            Count = Count + 1;

            if Count == NumOfNeighbours:

                break;

    return Numerator/Denominator;




def WeightedProfileSingleEntry(i,j,DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours,WeightedProfileThreshold):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    DrugBased   = DrugBasedPrediction(i,j,DDSimilarity,Interactions,NumOfNeighbours);

    TargetBased = TargetBasedPrediction(i,j,TTSimilarity,Interactions,NumOfNeighbours);

    Mean = (DrugBased + TargetBased)/2;

    if Mean >= WeightedProfileThreshold:

        return 1;

    return 0;


def WeightedProfile(DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours,WeightedProfileThreshold):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    NewInteractions = Interactions.copy();

    for i in range(0,NumOfDrugs):
        for j in range(0,NumOfTargets):

            #if Interactions.iloc[i,j] == 0:

            Pred = WeightedProfileSingleEntry(i,j,DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours, WeightedProfileThreshold);

            NewInteractions.iloc[i,j] = Pred;

    return NewInteractions;



def TTJaccardSimilarity(Interactions,T1,T2,ExcludedDrug):


    NumOfDrugs = Interactions.shape[0];

    Intersection = 0;

    Union = 0;

    for i in range(0,NumOfDrugs):

        if i != ExcludedDrug:

            Sum = Interactions.iloc[i,T1] + Interactions.iloc[i,T2];

            if Sum == 2:

                Intersection = Intersection + 1;

                Union = Union + 1;

            if Sum == 1:

                Union = Union + 1;

    if Union == 0:
        return 0;

    Similarity = Intersection / Union;

    return Similarity;



def DDJaccardSimilarity(Interactions,D1,D2,ExcludedTarget):


    NumOfTargets = Interactions.shape[1];

    Intersection = 0;

    Union = 0;

    for i in range(0,NumOfTargets):

        if i != ExcludedTarget:

            Sum = Interactions.iloc[D1,i] + Interactions.iloc[D2,i];

            if Sum == 2:

                Intersection = Intersection + 1;

                Union = Union + 1;

            if Sum == 1:

                Union = Union + 1;
                
                
    if Union == 0:
        return 0;

    Similarity = Intersection / Union;

    return Similarity;


def DDMatrixJaccardSimilarity(Interactions):


    NumOfDrugs = Interactions.shape[0];


    DDMatJaccardSimilarity = pd.DataFrame(index=range(0,NumOfDrugs),columns=range(0,NumOfDrugs));


    for i in range(0,NumOfDrugs):
        print(i+1," of ",NumOfDrugs,".");
        for j in range(i,NumOfDrugs):


            if i == j:
                DDMatJaccardSimilarity.iloc[i,j] = 1;
            else:
                DDMatJaccardSimilarity.iloc[i,j] = DDJaccardSimilarity(Interactions,i,j,-1);
                DDMatJaccardSimilarity.iloc[j,i] = DDMatJaccardSimilarity.iloc[i,j];
            
   

    return DDMatJaccardSimilarity;


def TTMatrixJaccardSimilarity(Interactions):


    NumOfTargets = Interactions.shape[1];


    TTMatJaccardSimilarity = pd.DataFrame(index=range(0,NumOfTargets),columns=range(0,NumOfTargets));


    for i in range(0,NumOfTargets):
        print(i+1," of ",NumOfTargets,".");
        for j in range(0,NumOfTargets):

            if i == j:
                TTMatJaccardSimilarity.iloc[i,j] = 1;
            else:
                TTMatJaccardSimilarity.iloc[i,j] = TTJaccardSimilarity(Interactions,i,j,-1);
                TTMatJaccardSimilarity.iloc[j,i] = TTMatJaccardSimilarity.iloc[i,j];
            
   

    return TTMatJaccardSimilarity;


def evaluation(DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours,WeightedProfileThreshold):
    
        
        scores = []
        for d, t in Interactions:
            score = WeightedProfileSingleEntry(d,t,DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours,WeightedProfileThreshold)      
            scores.append(score)
            
        
        prec, rec, thr = precision_recall_curve(Interactions, scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(Interactions, scores)
        auc_val = auc(fpr, tpr)
        
        #!!!!we should distinguish here between inverted and not inverted methods nDCGs!!!!
        return aupr_val, auc_val



#======================================================================


OriginalData = DataReadWrite.ReadOriginalKinase();


DDOriginalSimilarity = OriginalData["DDSimilarity"];


TTOriginalSimilarity = OriginalData["TTSimilarity"];


Interactions = OriginalData["Interactions"];


#Call this function once to calculate the Jaccard Similarities and write it to csv files 
#DataReadWrite.WriteJaccardKinase(DDMatrixJaccardSimilarity(Interactions),TTMatrixJaccardSimilarity(Interactions));


JaccardData = DataReadWrite.ReadJaccardKinase();

DDJaccardSimilarity = JaccardData["DDSimilarity"];

TTJaccardSimilarity = JaccardData["TTSimilarity"];

JaccardInteractions = JaccardData["Interactions"];




PredInteractions1 = WeightedProfile(DDOriginalSimilarity,TTOriginalSimilarity,Interactions,2,0.5);

PredInteractions2 = WeightedProfile(DDJaccardSimilarity,TTJaccardSimilarity,JaccardInteractions,2,0.5);


print(PredInteractions1);

print("-----------------");

print(PredInteractions2);







    


    

    





            
        

    




