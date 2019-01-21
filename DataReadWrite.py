import pandas as pd
import numpy as np
import os

def ReadOriginalKinase():

    DDSimilarity = pd.read_csv('Datasets\\Kinase\\Kinase_OriginalDD.txt', sep=" " , header = None);

    DDSimilarity = DDSimilarity.drop([0], axis=1);

    TTSimilarity = pd.read_csv('Datasets\\Kinase\\Kinase_OriginalTT.txt', sep=" " , header = None);

    TTSimilarity = TTSimilarity.drop([0], axis=1);

    Interactions = pd.read_csv('Datasets\\Kinase\\Kinase_Interactions.txt', sep=" " , header = None);

    Interactions = Interactions.drop([0], axis=1);

    return {"DDSimilarity":DDSimilarity, "TTSimilarity":TTSimilarity, "Interactions":Interactions};

def load_data_from_file(dataset, folder):
    with open(os.path.join(folder, dataset+"_admat_dgc.txt"), "r") as inf:
        inf.readline()
        int_array = [line.strip("\n").split()[1:] for line in inf]

    with open(os.path.join(folder, dataset+"_simmat_dc.txt"), "r") as inf:  # the drug similarity file
        inf.readline()
        drug_sim = [line.strip("\n").split()[1:] for line in inf]

    with open(os.path.join(folder, dataset+"_simmat_dg.txt"), "r") as inf:  # the target similarity file
        inf.readline()
        target_sim = [line.strip("\n").split()[1:] for line in inf]

    intMat = np.array(int_array, dtype=np.float64).T    # drug-target interaction matrix
    drugMat = np.array(drug_sim, dtype=np.float64)      # drug similarity matrix
    targetMat = np.array(target_sim, dtype=np.float64)  # target similarity matrix
    intMat = pd.DataFrame(intMat)
    drugMat = pd.DataFrame(drugMat)
    targetMat = pd.DataFrame(targetMat)
    return intMat, drugMat, targetMat

def WriteJaccard(DDMatrixJaccardSimilarityIntersection,DDMatrixJaccardSimilarityUnion,TTMatrixJaccardSimilarityIntersection,TTMatrixJaccardSimilarityUnion,Folder,Name):


    DDMatrixJaccardSimilarityIntersection.to_csv(Folder + "\\"+ Name +"_JaccardDDIntersection.csv",index=False);

    DDMatrixJaccardSimilarityUnion.to_csv(Folder + "\\"+ Name +"_JaccardDDUnion.csv",index=False);
    
    TTMatrixJaccardSimilarityIntersection.to_csv(Folder + "\\"+ Name +"_JaccardTTIntersection.csv",index=False);

    TTMatrixJaccardSimilarityUnion.to_csv(Folder + "\\"+ Name +"_JaccardTTUnion.csv",index=False);

    print("Done");


def ReadJaccardKinase():

    DDSimilarityIntersection = pd.read_csv("Datasets\\Kinase\\Kinase_JaccardDDIntersection.csv");

    DDSimilarityUnion = pd.read_csv("Datasets\\Kinase\\Kinase_JaccardDDUnion.csv");

    TTSimilarityIntersection = pd.read_csv("Datasets\\Kinase\\Kinase_JaccardTTIntersection.csv");

    TTSimilarityUnion = pd.read_csv("Datasets\\Kinase\\Kinase_JaccardTTUnion.csv");   

    Interactions = pd.read_csv('Datasets\\Kinase\\Kinase_Interactions.txt', sep=" " , header = None);

    Interactions = Interactions.drop([0], axis=1);

    return DDSimilarityIntersection,DDSimilarityUnion,TTSimilarityIntersection,TTSimilarityUnion,Interactions;

    
