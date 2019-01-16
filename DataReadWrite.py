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



def WriteJaccardKinase(DDJaccardSimilarity,TTJaccardSimilarity):


    DDJaccardSimilarity.to_csv("Datasets\\Kinase\\Kinase_JaccardDD.csv",index=False);

    TTJaccardSimilarity.to_csv("Datasets\\Kinase\\Kinase_JaccardTT.csv",index=False);

    print("Done");


def ReadJaccardKinase():

    DDSimilarity = pd.read_csv("Datasets\\Kinase\\Kinase_JaccardDD.csv");

    TTSimilarity = pd.read_csv("Datasets\\Kinase\\Kinase_JaccardTT.csv");

    Interactions = pd.read_csv('Datasets\\Kinase\\Kinase_Interactions.txt', sep=" " , header = None);

    Interactions = Interactions.drop([0], axis=1);

    return {"DDSimilarity":DDSimilarity, "TTSimilarity":TTSimilarity, "Interactions":Interactions};

    
