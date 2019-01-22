import pandas as pd
import numpy as np
import DataReadWrite
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import statistics
import math

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


    return Intersection,Union;


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
                
                
    
    return Intersection , Union;


def DDMatrixJaccardSimilarity(Interactions):


    NumOfDrugs = Interactions.shape[0];


    DDMatJaccardSimilarityIntersection = pd.DataFrame(index=range(0,NumOfDrugs),columns=range(0,NumOfDrugs));

    DDMatJaccardSimilarityUnion = pd.DataFrame(index=range(0,NumOfDrugs),columns=range(0,NumOfDrugs));


    for i in range(0,NumOfDrugs):
        print(i+1," of ",NumOfDrugs,".");
        for j in range(i,NumOfDrugs):


            if i == j:
                DDMatJaccardSimilarityIntersection.iloc[i,j] = 1;
                DDMatJaccardSimilarityUnion.iloc[i,j] = 1;
            else:

                Intersection , Union = DDJaccardSimilarity(Interactions,i,j,-1);

                DDMatJaccardSimilarityIntersection.iloc[i,j] = Intersection;

                DDMatJaccardSimilarityIntersection.iloc[j,i] = Intersection;


                DDMatJaccardSimilarityUnion.iloc[i,j] = Union;

                DDMatJaccardSimilarityUnion.iloc[j,i] = Union;
            
   

    return DDMatJaccardSimilarityIntersection,DDMatJaccardSimilarityUnion;



def TTMatrixJaccardSimilarity(Interactions):


    NumOfTargets = Interactions.shape[1];


    TTMatJaccardSimilarityIntersection = pd.DataFrame(index=range(0,NumOfTargets),columns=range(0,NumOfTargets));
    TTMatJaccardSimilarityUnion = pd.DataFrame(index=range(0,NumOfTargets),columns=range(0,NumOfTargets));


    for i in range(0,NumOfTargets):
        print(i+1," of ",NumOfTargets,".");
        for j in range(0,NumOfTargets):

            if i == j:
                TTMatJaccardSimilarityIntersection.iloc[i,j] = 1;
                TTMatJaccardSimilarityUnion.iloc[i,j] = 1;
                
            else:
                Intersection , Union = TTJaccardSimilarity(Interactions,i,j,-1);
                
                TTMatJaccardSimilarityIntersection.iloc[i,j] = Intersection;
                TTMatJaccardSimilarityIntersection.iloc[j,i] = Intersection;

                TTMatJaccardSimilarityUnion.iloc[i,j] = Union;
                TTMatJaccardSimilarityUnion.iloc[j,i] = Union;

    return TTMatJaccardSimilarityIntersection,TTMatJaccardSimilarityUnion;


def DrugBasedPrediction(i,j,DDSimilarityIntersection,DDSimilarityUnion,Interactions,NumOfNeighbours,Recalculate):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    Numerator  = 0;

    Denominator = 0;

    Sim = [];

    for k in range(0,NumOfDrugs):

        Intersection = DDSimilarityIntersection.iloc[i,k];

        Union = DDSimilarityUnion.iloc[i,k];

        if Recalculate == 1:

            if Interactions.iloc[i,j] == 1 and Interactions.iloc[k,j]==1 :

                Intersection = Intersection - 1;

            if Interactions.iloc[i,j] == 1 and Interactions.iloc[k,j]==0:

                Union = Union - 1;

        if Union <= 0:
            Sim.append(0);
        else:
            Sim.append((Intersection/Union));
            
        
    SortedIndices = np.argsort(Sim);

    SortedIndices = SortedIndices[::-1];

    Count = 0;

    for k in range(0, NumOfDrugs):

        CurrentIndex = SortedIndices[k];

        if CurrentIndex != i:

            Numerator = Numerator + Interactions.iloc[CurrentIndex,j]*Sim[CurrentIndex];

            Denominator = Denominator + Sim[CurrentIndex];

            Count = Count + 1;

            if Count == NumOfNeighbours:

                break;

    if Denominator == 0:
        return 0;
    
    return Numerator/Denominator;



def TargetBasedPrediction(i,j,TTSimilarityIntersection,TTSimilarityUnion,Interactions,NumOfNeighbours,Recalculate):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    Numerator  = 0;

    Denominator = 0;

    Sim = [];

    for k in range(0,NumOfTargets):

        Intersection = TTSimilarityIntersection.iloc[j,k];

        Union = TTSimilarityUnion.iloc[j,k];

        if Recalculate==1:

            if Interactions.iloc[i,j] == 1 and Interactions.iloc[i,k] == 1:

                Intersection = Intersection - 1;

            if Interactions.iloc[i,j] == 1 and Interactions.iloc[i,k]== 0:

                Union = Union - 1;

        if Union <= 0:
            Sim.append(0);
        else:
            Sim.append((Intersection/Union));


    SortedIndices = np.argsort(Sim);

    SortedIndices = SortedIndices[::-1];

    Count = 0;

    for k in range(0, NumOfTargets):

        CurrentIndex = SortedIndices[k];

        if CurrentIndex != j:

            Numerator = Numerator + Interactions.iloc[i,CurrentIndex]*Sim[CurrentIndex];

            Denominator = Denominator + Sim[CurrentIndex];

            Count = Count + 1;

            if Count == NumOfNeighbours:

                break;

    if Denominator == 0:
        return 0;

    return Numerator/Denominator;



def WeightedProfileSingleEntry(i,j,DDSimilarityIntersection,DDSimilarityUnion,TTSimilarityIntersection,TTSimilarityUnion,Interactions,NumOfNeighbours,Recalculate):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    DrugBased   = DrugBasedPrediction(i,j,DDSimilarityIntersection,DDSimilarityUnion,Interactions,NumOfNeighbours,Recalculate);

    TargetBased = TargetBasedPrediction(i,j,TTSimilarityIntersection,TTSimilarityUnion,Interactions,NumOfNeighbours,Recalculate);

    Mean = (DrugBased + TargetBased)/2;

    return Mean;



def WeightedProfile(DDSimilarityIntersection,DDSimilarityUnion,TTSimilarityIntersection,TTSimilarityUnion,Interactions,NumOfNeighbours,Recalculate):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    NewInteractions = Interactions.copy();

    for i in range(0,NumOfDrugs):
        print("Predicting ..",i+1,NumOfDrugs);
        for j in range(0,NumOfTargets):

            Pred = WeightedProfileSingleEntry(i,j,DDSimilarityIntersection,DDSimilarityUnion,TTSimilarityIntersection,TTSimilarityUnion,Interactions,NumOfNeighbours,Recalculate);

            NewInteractions.iloc[i,j] = Pred;

    return NewInteractions;


def Evaluation(Interactions,NewInteractions):

    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    TruelLabels = []

    Scores = []

    for i in range(0,NumOfDrugs):
        for j in range(0,NumOfTargets):
            
            TruelLabels.append(Interactions.iloc[i,j]);

            Score = NewInteractions.iloc[i,j];
            
            Scores.append(Score)
            
        
    prec, rec, thr = precision_recall_curve(TruelLabels, Scores)
        
    aupr_val = auc(rec, prec)

    fpr, tpr, thr = roc_curve(TruelLabels, Scores)

    auc_val = auc(fpr, tpr)
        

    return auc_val,aupr_val



def Run(DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours,Recalculate,Dataset,Gen):


    if Gen == 1:

        I1,U1 = DDMatrixJaccardSimilarity(Interactions);

        I2,U2 = TTMatrixJaccardSimilarity(Interactions);

        DataReadWrite.WriteJaccard(I1,U1,I2,U2,"Datasets",Dataset);

    I1,U1,I2,U2 = DataReadWrite.ReadJaccard("Datasets",Dataset);

    Predictions = WeightedProfile(I1,U1,I2,U2,Interactions,NumOfNeighbours,Recalculate);

    print("Evaluating ...")

    print(Predictions)

    AUC , AUPR = Evaluation(Interactions,Predictions);

    print("Done");

    print("AUC : ", AUC);

    print("AUPR : ", AUPR);

