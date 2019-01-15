import pandas as pd
import numpy as np




def ReadKinase():

    DDSimilarity = pd.read_csv('Datasets\\Kinase_DD.txt', sep=" " , header = None);

    DDSimilarity = DDSimilarity.drop([0], axis=1);

    TTSimilarity = pd.read_csv('Datasets\\Kinase_TT.txt', sep=" " , header = None);

    TTSimilarity = TTSimilarity.drop([0], axis=1);

    Interactions = pd.read_csv('Datasets\\Kinase_Interactions.txt', sep=" " , header = None);

    Interactions = Interactions.drop([0], axis=1);

    return {"DDSimilarity":DDSimilarity, "TTSimilarity":TTSimilarity, "Interactions":Interactions};



Data = ReadKinase();


_DDSimilarity = Data["DDSimilarity"];


_TTSimilarity = Data["TTSimilarity"];


_Interactions = Data["Interactions"];


_NumOfDrugs = _Interactions.shape[0];


_NumOfTargets = _Interactions.shape[1];


_NumOfNeighbours = 2;


def DrugBasedPrediction(i,j,DDSimilarity,Interactions,NumOfDrugs,NumOfTargets,NumOfNeighbours):

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



def TargetBasedPrediction(i,j,TTSimilarity,Interactions,NumOfDrugs,NumOfTargets,NumOfNeighbours):

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

def SimpleWeightedProfileSingleEntry(i,j,DDSimilarity,TTSimilarity,Interactions,NumOfDrugs,NumOfTargets,NumOfNeighbours):


    DrugBased   = DrugBasedPrediction(i,j,DDSimilarity,Interactions,NumOfDrugs,NumOfTargets,NumOfNeighbours);

    TargetBased = TargetBasedPrediction(i,j,TTSimilarity,Interactions,NumOfDrugs,NumOfTargets,NumOfNeighbours);

    Mean = (DrugBased + TargetBased)/2;

    if Mean >= 0.5:

        return 1;

    return 0;


def SimpleWeightedProfile(DDSimilarity,TTSimilarity,Interactions,NumOfDrugs,NumOfTargets,NumOfNeighbours):

    NewInteractions = _Interactions.copy();

    for i in range(0,NumOfDrugs):
        for j in range(0,NumOfTargets):

            if Interactions.iloc[i,j] == 0:

                Pred = SimpleWeightedProfileSingleEntry(i,j,DDSimilarity,TTSimilarity,Interactions,NumOfDrugs,NumOfTargets,NumOfNeighbours);

                NewInteractions.iloc[i,j] = Pred;

    return NewInteractions;


PredInteractionsMatrix = SimpleWeightedProfile(_DDSimilarity,_TTSimilarity,_Interactions,_NumOfDrugs,_NumOfTargets,_NumOfNeighbours);


def LeaveOneOutCrossValidation(DDSimilarity,TTSimilarity,PredInteractionsMatrix,NumOfDrugs,NumOfTargets,NumOfNeighbours):


    Count = 0;

    for i in range(0,NumOfDrugs):
        for j in range(0,NumOfTargets):

            Pred = SimpleWeightedProfileSingleEntry(i,j,DDSimilarity,TTSimilarity,PredInteractionsMatrix,NumOfDrugs,NumOfTargets,NumOfNeighbours);

            if PredInteractionsMatrix.iloc[i,j] == Pred:

                Count = Count + 1;


    return Count;


Count = LeaveOneOutCrossValidation(_DDSimilarity,_TTSimilarity,PredInteractionsMatrix,_NumOfDrugs,_NumOfTargets,_NumOfNeighbours);

print(Count);


            
        

    




