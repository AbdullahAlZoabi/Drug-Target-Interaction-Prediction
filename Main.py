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




def SimpleWeightedProfileSingleEntry(i,j,DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours,SimpleWeightedProfileThreshold):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    DrugBased   = DrugBasedPrediction(i,j,DDSimilarity,Interactions,NumOfNeighbours);

    TargetBased = TargetBasedPrediction(i,j,TTSimilarity,Interactions,NumOfNeighbours);

    Mean = (DrugBased + TargetBased)/2;

    if Mean >= SimpleWeightedProfileThreshold:

        return 1;

    return 0;


def SimpleWeightedProfile(DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours,SimpleWeightedProfileThreshold):


    NumOfDrugs = Interactions.shape[0];

    NumOfTargets = Interactions.shape[1];

    NewInteractions = Interactions.copy();

    for i in range(0,NumOfDrugs):
        for j in range(0,NumOfTargets):

            #if Interactions.iloc[i,j] == 0:

            Pred = SimpleWeightedProfileSingleEntry(i,j,DDSimilarity,TTSimilarity,Interactions,NumOfNeighbours,SimpleWeightedProfileThreshold);

            NewInteractions.iloc[i,j] = Pred;

    return NewInteractions;





PredInteractionsMatrix = SimpleWeightedProfile(_DDSimilarity,_TTSimilarity,_Interactions,2,0.5);



print(PredInteractionsMatrix)


            
        

    




