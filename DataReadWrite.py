import pandas as pd
import numpy as np

def ReadOriginalKinase():

    DDSimilarity = pd.read_csv('Datasets\\Kinase\\Kinase_OriginalDD.txt', sep=" " , header = None);

    DDSimilarity = DDSimilarity.drop([0], axis=1);

    TTSimilarity = pd.read_csv('Datasets\\Kinase\\Kinase_OriginalTT.txt', sep=" " , header = None);

    TTSimilarity = TTSimilarity.drop([0], axis=1);

    Interactions = pd.read_csv('Datasets\\Kinase\\Kinase_Interactions.txt', sep=" " , header = None);

    Interactions = Interactions.drop([0], axis=1);

    return {"DDSimilarity":DDSimilarity, "TTSimilarity":TTSimilarity, "Interactions":Interactions};



def WriteJaccardKinase(DDMatrixJaccardSimilarityIntersection,DDMatrixJaccardSimilarityUnion,TTMatrixJaccardSimilarityIntersection,TTMatrixJaccardSimilarityUnion):


    DDMatrixJaccardSimilarityIntersection.to_csv("Datasets\\Kinase\\Kinase_JaccardDDIntersection.csv",index=False);

    DDMatrixJaccardSimilarityUnion.to_csv("Datasets\\Kinase\\Kinase_JaccardDDUnion.csv",index=False);
    
    TTMatrixJaccardSimilarityIntersection.to_csv("Datasets\\Kinase\\Kinase_JaccardTTIntersection.csv",index=False);

    TTMatrixJaccardSimilarityUnion.to_csv("Datasets\\Kinase\\Kinase_JaccardTTUnion.csv",index=False);

    print("Done");


def ReadJaccardKinase():

    DDSimilarityIntersection = pd.read_csv("Datasets\\Kinase\\Kinase_JaccardDDIntersection.csv");

    DDSimilarityUnion = pd.read_csv("Datasets\\Kinase\\Kinase_JaccardDDUnion.csv");

    TTSimilarityIntersection = pd.read_csv("Datasets\\Kinase\\Kinase_JaccardTTIntersection.csv");

    TTSimilarityUnion = pd.read_csv("Datasets\\Kinase\\Kinase_JaccardTTUnion.csv");   

    Interactions = pd.read_csv('Datasets\\Kinase\\Kinase_Interactions.txt', sep=" " , header = None);

    Interactions = Interactions.drop([0], axis=1);

    return DDSimilarityIntersection,DDSimilarityUnion,TTSimilarityIntersection,TTSimilarityUnion,Interactions;

    
