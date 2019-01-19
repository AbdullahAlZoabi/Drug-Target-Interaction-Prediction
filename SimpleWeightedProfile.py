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




def WeightedProfileSingleEntry(i,j,DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    DrugBased   = DrugBasedPrediction(i,j,DDSimilarity,Interactions,NumOfNeighbours);

    TargetBased = TargetBasedPrediction(i,j,TTSimilarity,Interactions,NumOfNeighbours);

    Mean = (DrugBased + TargetBased)/2;

    return Mean;




def WeightedProfile(DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    NewInteractions = Interactions.copy();

    for i in range(0,NumOfDrugs):
        print("Predicting ..",i+1,NumOfDrugs);
        for j in range(0,NumOfTargets):

            Pred = WeightedProfileSingleEntry(i,j,DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours);

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






























