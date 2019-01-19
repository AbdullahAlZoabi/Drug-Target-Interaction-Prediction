import pandas as pd
import numpy as np
import DataReadWrite
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import statistics
import math


def DrugsWeightingForTarget(T,DDSimilarity,Interactions,NumOfNeighbours):


    NumOfDrugs = Interactions.shape[0];

    BadNeighbours = [0] * NumOfDrugs;

    for i in range(NumOfDrugs):

        SortedIndices = np.argsort(-DDSimilarity.iloc[i,:]);

        Count = 0;

        for j in range(0,NumOfDrugs):

            CurrentIndex = SortedIndices.iloc[j,];

            if CurrentIndex != i:

                if Interactions.iloc[i,T] != Interactions.iloc[CurrentIndex,T]:

                    BadNeighbours[CurrentIndex] = BadNeighbours[CurrentIndex]+ 1;

                Count = Count + 1;

            if Count == NumOfNeighbours:
                
                break;


    Mean = statistics.mean(BadNeighbours);

    StDev = statistics.stdev(BadNeighbours);

    for i in range(NumOfDrugs):
        
        if Mean == 0:
            H = 0;
        else:
            H = (BadNeighbours[i] - Mean)/StDev;

        W = math.exp(-H);

        BadNeighbours[i] = W;

    return BadNeighbours;



def DrugsWeighting(DDSimilarity,Interactions,NumOfNeighbours):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    WeightesMatrix = pd.DataFrame(index=range(0,NumOfTargets),columns=range(0,NumOfDrugs));

    for i in range(0,NumOfTargets):

        print("Drugs Weighting.. ",i+1,"Of",NumOfTargets)

        ForSingleTarget = DrugsWeightingForTarget(i,DDSimilarity,Interactions,NumOfNeighbours);

        for j in range(0,NumOfDrugs):

            WeightesMatrix.iloc[i,j] = ForSingleTarget[j];


    return WeightesMatrix;




def TargetsWeightingForDrug(D,TTSimilarity,Interactions,NumOfNeighbours):


    NumOfTargets = Interactions.shape[1];

    BadNeighbours = [0] * NumOfTargets;

    for i in range(NumOfTargets):

        SortedIndices = np.argsort(-TTSimilarity.iloc[i,:]);

        Count = 0;

        for j in range(0,NumOfTargets):

            CurrentIndex = SortedIndices.iloc[j,];

            if CurrentIndex != i:

                if Interactions.iloc[D,i] != Interactions.iloc[D,CurrentIndex]:

                    BadNeighbours[CurrentIndex] = BadNeighbours[CurrentIndex]+ 1;

                Count = Count + 1;

            if Count == NumOfNeighbours:
                
                break;


    Mean = statistics.mean(BadNeighbours);

    StDev = statistics.stdev(BadNeighbours);

    for i in range(NumOfTargets):
        
        if Mean == 0:
            H = 0;
        else:
            H = (BadNeighbours[i] - Mean)/StDev;

        W = math.exp(-H);

        BadNeighbours[i] = W;

    return BadNeighbours;



def TargetsWeighting(TTSimilarity,Interactions,NumOfNeighbours):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    WeightesMatrix = pd.DataFrame(index=range(0,NumOfDrugs),columns=range(0,NumOfTargets));

    for i in range(0,NumOfDrugs):

        print("Targets Weighting.. ",i+1,"Of",NumOfDrugs)

        ForSingleDrug = TargetsWeightingForDrug(i,TTSimilarity,Interactions,NumOfNeighbours);

        for j in range(0,NumOfTargets):

            WeightesMatrix.iloc[i,j] = ForSingleDrug[j];


    return WeightesMatrix;


def DrugBasedPrediction(i,j,DDSimilarity,Interactions,NumOfNeighbours,Weightes):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    Numerator  = 0;

    Denominator = 0;

    SortedIndices = np.argsort(-DDSimilarity.iloc[i,:]);

    Count = 0;

    for k in range(0, NumOfDrugs):

        CurrentIndex = SortedIndices.iloc[k,];

        if CurrentIndex != i:

            Numerator = Numerator + Interactions.iloc[CurrentIndex,j]*DDSimilarity.iloc[i,CurrentIndex]*Weightes.iloc[j,CurrentIndex];

            Denominator = Denominator + DDSimilarity.iloc[i,CurrentIndex]*Weightes.iloc[j,CurrentIndex];

            Count = Count + 1;

            if Count == NumOfNeighbours:

                break;

    return Numerator/Denominator;




def TargetBasedPrediction(i,j,TTSimilarity,Interactions,NumOfNeighbours,Weightes):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    Numerator  = 0;

    Denominator = 0;

    SortedIndices = np.argsort(-TTSimilarity.iloc[j,:]);

    Count = 0;

    for k in range(0, NumOfTargets):

        CurrentIndex = SortedIndices.iloc[k,];

        if CurrentIndex != j:

            Numerator = Numerator + Interactions.iloc[i,CurrentIndex]*TTSimilarity.iloc[j,CurrentIndex]*Weightes.iloc[i,CurrentIndex];

            Denominator = Denominator + TTSimilarity.iloc[j,CurrentIndex]*Weightes.iloc[i,CurrentIndex];

            Count = Count + 1;

            if Count == NumOfNeighbours:

                break;

    return Numerator/Denominator;




def WeightedProfileSingleEntry(i,j,DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours,WeightesDrugs,WeightesTargets):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    DrugBased   = DrugBasedPrediction(i,j,DDSimilarity,Interactions,NumOfNeighbours,WeightesDrugs);

    TargetBased = TargetBasedPrediction(i,j,TTSimilarity,Interactions,NumOfNeighbours,WeightesTargets);

    Mean = (DrugBased + TargetBased)/2;

    return Mean;



def WeightedProfile(DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    NewInteractions = Interactions.copy();

    WeightesDrugs = DrugsWeighting(DDSimilarity,Interactions,NumOfNeighbours);

    WeightesTargets = TargetsWeighting(TTSimilarity,Interactions,NumOfNeighbours);

    for i in range(0,NumOfDrugs):
        print("Predicting .. ",i+1,"Of",NumOfDrugs)
        for j in range(0,NumOfTargets):

            Pred = WeightedProfileSingleEntry(i,j,DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours,WeightesDrugs,WeightesTargets);

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
            
            if Score!=Score:
                Scores.append(0)
            else:
                Scores.append(Score)
            
        
        prec, rec, thr = precision_recall_curve(TruelLabels, Scores)
        
        aupr_val = auc(rec, prec)

        fpr, tpr, thr = roc_curve(TruelLabels, Scores)

        auc_val = auc(fpr, tpr)
        

        return auc_val,aupr_val



def Run(DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours):



    Predictions = WeightedProfile(DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours);

    print("Evaluating ...")

    AUC , AUPR = Evaluation(Interactions,Predictions);

    print("Done");

    print("AUC : ", AUC);

    print("AUPR : ", AUPR);

    
    

    

















