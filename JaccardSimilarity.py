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





