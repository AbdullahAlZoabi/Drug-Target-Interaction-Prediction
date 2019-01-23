import pandas as pd
import numpy as np
import DataReadWrite
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import statistics
import math
import HubnessAware
import SimpleWeightedProfile
import JaccardSimilarity


Mthods = ["SimpleWP", "H-awareWP","JaccardWP","JaccardWPWithRecalculating"];

Datasets = ["Enzyme","Ion Channels","GPCR","NR","Kinase"];

#--------------------------------------------------------------------------- Change this section only to run an experiment

GenerateJaccardSimilarity = 0; # set 1 to generate. but all the data is already generated

NumberOfNeighbours = 3;

SelectedDataset = "Kinase";

SelectedMethod = "SimpleWP";


#--------------------------------------------------------------------------
print("Number Of Neighbours : ",NumberOfNeighbours);

print("Dataset : ",SelectedDataset);

print("Method : ",SelectedMethod);


#---------------------------------------------------------------------------


if SelectedDataset == "Kinase":

    Data = DataReadWrite.ReadOriginalKinase();

    DDSimilarity = Data["DDSimilarity"];

    TTSimilarity = Data["TTSimilarity"];

    Interactions = Data["Interactions"];

    if SelectedMethod == "SimpleWP":
        SimpleWeightedProfile.Run(DDSimilarity,TTSimilarity,Interactions,NumberOfNeighbours);
    if SelectedMethod == "H-awareWP":
        HubnessAware.Run(DDSimilarity,TTSimilarity,Interactions,NumberOfNeighbours);
    if SelectedMethod == "JaccardWP":
        JaccardSimilarity.Run(DDSimilarity,TTSimilarity,Interactions,NumberOfNeighbours,0,"Kinase",GenerateJaccardSimilarity)
    if SelectedMethod == "JaccardWPWithRecalculating":
        JaccardSimilarity.Run(DDSimilarity,TTSimilarity,Interactions,NumberOfNeighbours,1,"Kinase",GenerateJaccardSimilarity)
    



if SelectedDataset == "NR":


    Interactions, DDSimilarity ,TTSimilarity  = DataReadWrite.load_data_from_file("nr", 'datasets')

    if SelectedMethod == "SimpleWP":
        SimpleWeightedProfile.Run(DDSimilarity,TTSimilarity,Interactions,NumberOfNeighbours);
    if SelectedMethod == "H-awareWP":
        HubnessAware.Run(DDSimilarity,TTSimilarity,Interactions,NumberOfNeighbours);
    if SelectedMethod == "JaccardWP":
        JaccardSimilarity.Run(DDSimilarity,TTSimilarity,Interactions,NumberOfNeighbours,0,"nr",GenerateJaccardSimilarity)
    if SelectedMethod == "JaccardWPWithRecalculating":
        JaccardSimilarity.Run(DDSimilarity,TTSimilarity,Interactions,NumberOfNeighbours,1,"nr",GenerateJaccardSimilarity)



if SelectedDataset == "GPCR":


    Interactions, DDSimilarity ,TTSimilarity  = DataReadWrite.load_data_from_file("gpcr", 'datasets')

    if SelectedMethod == "SimpleWP":
        SimpleWeightedProfile.Run(DDSimilarity,TTSimilarity,Interactions,NumberOfNeighbours);
    if SelectedMethod == "H-awareWP":
        HubnessAware.Run(DDSimilarity,TTSimilarity,Interactions,NumberOfNeighbours);
    if SelectedMethod == "JaccardWP":
        JaccardSimilarity.Run(DDSimilarity,TTSimilarity,Interactions,NumberOfNeighbours,0,"gpcr",GenerateJaccardSimilarity)
    if SelectedMethod == "JaccardWPWithRecalculating":
        JaccardSimilarity.Run(DDSimilarity,TTSimilarity,Interactions,NumberOfNeighbours,1,"gpcr",GenerateJaccardSimilarity)



if SelectedDataset == "Ion Channels":


    Interactions, DDSimilarity ,TTSimilarity  = DataReadWrite.load_data_from_file("ic", 'datasets')

    if SelectedMethod == "SimpleWP":
        SimpleWeightedProfile.Run(DDSimilarity,TTSimilarity,Interactions,NumberOfNeighbours);
    if SelectedMethod == "H-awareWP":
        HubnessAware.Run(DDSimilarity,TTSimilarity,Interactions,NumberOfNeighbours);
    if SelectedMethod == "JaccardWP":
        JaccardSimilarity.Run(DDSimilarity,TTSimilarity,Interactions,NumberOfNeighbours,0,"ic",GenerateJaccardSimilarity)
    if SelectedMethod == "JaccardWPWithRecalculating":
        JaccardSimilarity.Run(DDSimilarity,TTSimilarity,Interactions,NumberOfNeighbours,1,"ic",GenerateJaccardSimilarity)




if SelectedDataset == "Enzyme":


    Interactions, DDSimilarity ,TTSimilarity  = DataReadWrite.load_data_from_file("e", 'datasets')

    if SelectedMethod == "SimpleWP":
        SimpleWeightedProfile.Run(DDSimilarity,TTSimilarity,Interactions,NumberOfNeighbours);
    if SelectedMethod == "H-awareWP":
        HubnessAware.Run(DDSimilarity,TTSimilarity,Interactions,NumberOfNeighbours);
    if SelectedMethod == "JaccardWP":
        JaccardSimilarity.Run(DDSimilarity,TTSimilarity,Interactions,NumberOfNeighbours,0,"e",GenerateJaccardSimilarity)
    if SelectedMethod == "JaccardWPWithRecalculating":
        JaccardSimilarity.Run(DDSimilarity,TTSimilarity,Interactions,NumberOfNeighbours,1,"e",GenerateJaccardSimilarity)





            
        

    




