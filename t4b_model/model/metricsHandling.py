import pandas as pd
import numpy as np

def calculateAEC(csvFileName: str, expecteConfidence: list = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50], decimalPoints: int = 2):

    df = pd.read_csv(csvFileName)

    df.iloc[:, 1:] = df.iloc[:, 1:].round(decimalPoints)

    expected_list = expecteConfidence

    columnNames = list(df.columns.values)
    
    i = 0
    for i in range(len(expected_list)):
        df[columnNames[i+3]] = (df[columnNames[i+3]] - expected_list[i]) * 100
        df.rename(columns={columnNames[i+3]: ("AEC " + str((expecteConfidence[i]*100)) + " [%]")}, inplace=True)
        i = i + 1
    return df

def renameSpacesSensor(df, newSpacenames: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], existingNames: list = ["007A", "008A", "011A", "012A", "013A", "015A", "020A", "020B", "029A", "031A", "033A", "035A"]):
    if len(newSpacenames) == len(existingNames):
        i = 0
        for i in range(len(existingNames)):
            df["ID"] = df["ID"].apply(lambda x: str(newSpacenames[i]) if existingNames[i] in x else x)
            i = i + 1
        
        return df
    else:
        print('The two lists containing the existing names and the new space names, must have the same length')

def metricFileProcessing(csvFileName: str, decimalPoints: int = 2, dropColumnNumbers: int = 5):
    df = calculateAEC(csvFileName, decimalPoints=decimalPoints)
    df = renameSpacesSensor(df)
    df = df.drop(df.columns[-dropColumnNumbers:], axis=1)

    df.iloc[:, 1:] = df.iloc[:, 1:].round(decimalPoints)
    df.to_csv("AEC_"+csvFileName, index=False)
    
    print(df)

metricFileProcessing("test2.csv")


