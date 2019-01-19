import pandas as pd
import numpy as np
import DataReadWrite
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import statistics
import math


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


def IDrugBasedPrediction(i,j,DDSimilarity,Interactions,NumOfNeighbours,W):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    Numerator  = 0;

    Denominator = 0;

    SortedIndices = np.argsort(-DDSimilarity.iloc[i,:]);

    Count = 0;

    for k in range(0, NumOfDrugs):

        CurrentIndex = SortedIndices.iloc[k,];

        if CurrentIndex != i:

            Numerator = Numerator + Interactions.iloc[CurrentIndex,j]*DDSimilarity.iloc[i,CurrentIndex]*W.iloc[j,CurrentIndex];

            Denominator = Denominator + DDSimilarity.iloc[i,CurrentIndex]*W.iloc[j,CurrentIndex];

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


def ITargetBasedPrediction(i,j,TTSimilarity,Interactions,NumOfNeighbours,W):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    Numerator  = 0;

    Denominator = 0;

    SortedIndices = np.argsort(-TTSimilarity.iloc[j,:]);

    Count = 0;

    for k in range(0, NumOfTargets):

        CurrentIndex = SortedIndices.iloc[k,];

        if CurrentIndex != j:

            Numerator = Numerator + Interactions.iloc[i,CurrentIndex]*TTSimilarity.iloc[j,CurrentIndex]*W.iloc[i,CurrentIndex];

            Denominator = Denominator + TTSimilarity.iloc[j,CurrentIndex];

            Count = Count + 1;

            if Count == NumOfNeighbours:

                break;

    return Numerator/Denominator;





def IWeightedProfileSingleEntry(i,j,DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours,W1,W2):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    DrugBased   = IDrugBasedPrediction(i,j,DDSimilarity,Interactions,NumOfNeighbours,W1);

    TargetBased = ITargetBasedPrediction(i,j,TTSimilarity,Interactions,NumOfNeighbours,W2);

    Mean = (DrugBased + TargetBased)/2;

    return Mean;

       
def WeightedProfile(DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    NewInteractions = Interactions.copy();

    for i in range(0,NumOfDrugs):
        for j in range(0,NumOfTargets):

            Pred = WeightedProfileSingleEntry(i,j,DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours);

            NewInteractions.iloc[i,j] = Pred;

    return NewInteractions;



def evaluation(DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours):

    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    true_labels = []

    scores = []

    for i in range(0,NumOfDrugs):
        for j in range(0,NumOfTargets):
            
            label = Interactions.iloc[i,j]

            true_labels.append(label)

            score = WeightedProfileSingleEntry(i,j,DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours)

            if score!=score:
                scores.append(0)
            else:
                scores.append(score)
            
        
        prec, rec, thr = precision_recall_curve(true_labels, scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(true_labels, scores)
        auc_val = auc(fpr, tpr)
        
        #!!!!we should distinguish here between inverted and not inverted methods nDCGs!!!!
        return aupr_val, auc_val



def Ievaluation(DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours):

    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    true_labels = []

    scores = []

    W1 = AllDrugsWeighting(DDSimilarity,Interactions,NumOfNeighbours);
    W2 = AllTargetsWeighting(TTSimilarity,Interactions,NumOfNeighbours);

    for i in range(0,NumOfDrugs):
        for j in range(0,NumOfTargets):
            
            label = Interactions.iloc[i,j]

            true_labels.append(label)

            score = IWeightedProfileSingleEntry(i,j,DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours,W1,W2)

            if score!=score:
                scores.append(0)
            else:
                scores.append(score)
            
        
        prec, rec, thr = precision_recall_curve(true_labels, scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(true_labels, scores)
        auc_val = auc(fpr, tpr)
        
        #!!!!we should distinguish here between inverted and not inverted methods nDCGs!!!!
        return aupr_val, auc_val
    


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


def SingleDrugsRowWeighting(T,DDSimilarity,Interactions,NumOfNeighbours):


    NumOfDrugs = Interactions.shape[0];

    BN = [0] * NumOfDrugs;

    for i in range(NumOfDrugs):

        SortedIndices = np.argsort(-DDSimilarity.iloc[i,:]);

        Count = 0;

        for j in range(0,NumOfDrugs):

            CurrentIndex = SortedIndices.iloc[j,];

            if CurrentIndex != i:

                if Interactions.iloc[i,T] != Interactions.iloc[CurrentIndex,T]:

                    BN[CurrentIndex] = BN[CurrentIndex]+ 1;

                Count = Count + 1;

            if Count == NumOfNeighbours:
                
                break;


    Mean = statistics.mean(BN);

    StDev = statistics.stdev(BN);

    for i in range(NumOfDrugs):
        
        if StDev == 0:
            H = 0;
        else:
            H = (BN[i] - Mean)/StDev;

        W = math.exp(-H);

        BN[i] = W;

    return(BN);



def AllDrugsWeighting(DDSimilarity,Interactions,NumOfNeighbours):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    DrugsWeightes = pd.DataFrame(index=range(0,NumOfTargets),columns=range(0,NumOfDrugs));

    for i in range(0,NumOfTargets):

        Row = SingleDrugsRowWeighting(i,DDSimilarity,Interactions,NumOfNeighbours);

        for j in range(0,NumOfDrugs):

            DrugsWeightes.iloc[i,j] = Row[j];


    return DrugsWeightes;

    

def SingleTargetsColWeighting(D,TTSimilarity,Interactions,NumOfNeighbours):


    NumOfTargets = Interactions.shape[1];

    BN = [0] * NumOfTargets;

    for i in range(NumOfTargets):

        SortedIndices = np.argsort(-TTSimilarity.iloc[i,:]);

        Count = 0;

        for j in range(0,NumOfTargets):

            CurrentIndex = SortedIndices.iloc[j,];

            if CurrentIndex != i:

                if Interactions.iloc[D,i] != Interactions.iloc[D,CurrentIndex]:

                    BN[CurrentIndex] = BN[CurrentIndex]+ 1;

                Count = Count + 1;

            if Count == NumOfNeighbours:
                
                break;


    Mean = statistics.mean(BN);

    StDev = statistics.stdev(BN);

    for i in range(NumOfTargets):
        
        if StDev == 0:
            H = 0;
        else:
            H = (BN[i] - Mean)/StDev;

        W = math.exp(-H);

        BN[i] = W;

    return(BN);



def AllTargetsWeighting(TTSimilarity,Interactions,NumOfNeighbours):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    TargetsWeightes = pd.DataFrame(index=range(0,NumOfDrugs),columns=range(0,NumOfTargets));

    for i in range(0,NumOfDrugs):

        Col = SingleTargetsColWeighting(i,TTSimilarity,Interactions,NumOfNeighbours);

        for j in range(0,NumOfTargets):

            TargetsWeightes.iloc[i,j] = Col[j];


    return TargetsWeightes;





    









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




#PredInteractions1 = WeightedProfile(DDOriginalSimilarity,TTOriginalSimilarity,Interactions,2,0.5);

#PredInteractions2 = WeightedProfile(DDJaccardSimilarity,TTJaccardSimilarity,JaccardInteractions,2,0.5);


#print(PredInteractions1);

print("-----------------");

#print(PredInteractions2);



#test evaluation
aupr, auc = Ievaluation(DDJaccardSimilarity, TTJaccardSimilarity, JaccardInteractions,2)

print(auc)
print(aupr)




#print(AllTargetsWeighting(TTOriginalSimilarity,Interactions,2));



    


    

    





            
        

    




