'''
Created on Jul 10, 2019

Create a matrix that can be used in Conv1d and Conv2d and "special layer" ops

@author: cmihaigabriel@gmail.com
@version: v2.1

@change: v1.0 used in 2019 SI IJCV
@change: v2.0 used in 2020 ICMR
@change: v2.1 used in 2020 ECCV
'''
import numpy
import scipy, scipy.stats.stats
from collections import namedtuple

class SimMatrixCreator ():
    
    def getMatrix_CF32_4N4S4C(self, trainmatrix, testmatrix):
        (htrain, wtrain) = trainmatrix.shape
        (htest, wtest) = testmatrix.shape
        
        newtrainmatrix = numpy.zeros((htrain, wtrain * 3, 3, 3))
        newtestmatrix = numpy.zeros((htest, wtest * 3, 3, 3))
        
        for i in range(0, wtrain):
            corevector = trainmatrix[:, i]
            rlist = self.createPearsonRList(corevector, trainmatrix, i);
            
            for j in range(0, htrain):
                newtrainmatrix[j][3*i + 0][0][1] = trainmatrix[j][rlist[0][0]]
                newtrainmatrix[j][3*i + 0][1][1] = 0
                newtrainmatrix[j][3*i + 0][2][1] = trainmatrix[j][rlist[1][0]]
                newtrainmatrix[j][3*i + 1][0][1] = 0
                newtrainmatrix[j][3*i + 1][1][1] = trainmatrix[j][i]
                newtrainmatrix[j][3*i + 1][2][1] = trainmatrix[j][rlist[2][0]]
                newtrainmatrix[j][3*i + 2][0][1] = 0
                newtrainmatrix[j][3*i + 2][1][1] = trainmatrix[j][rlist[3][0]]
                newtrainmatrix[j][3*i + 2][2][1] = 0
                
                newtrainmatrix[j][3*i + 0][0][2] = rlist[0][1]
                newtrainmatrix[j][3*i + 0][1][2] = 0
                newtrainmatrix[j][3*i + 0][2][2] = rlist[1][1]
                newtrainmatrix[j][3*i + 1][0][2] = 0
                newtrainmatrix[j][3*i + 1][1][2] = 1
                newtrainmatrix[j][3*i + 1][2][2] = rlist[2][1]
                newtrainmatrix[j][3*i + 2][0][2] = 0
                newtrainmatrix[j][3*i + 2][1][2] = rlist[3][1]
                newtrainmatrix[j][3*i + 2][2][2] = 0
                
                newtrainmatrix[j][3*i + 0][0][0] = trainmatrix[j][i]
                newtrainmatrix[j][3*i + 0][1][0] = 0
                newtrainmatrix[j][3*i + 0][2][0] = trainmatrix[j][i]
                newtrainmatrix[j][3*i + 1][0][0] = 0
                newtrainmatrix[j][3*i + 1][1][0] = trainmatrix[j][i]
                newtrainmatrix[j][3*i + 1][2][0] = trainmatrix[j][i]
                newtrainmatrix[j][3*i + 2][0][0] = 0
                newtrainmatrix[j][3*i + 2][1][0] = trainmatrix[j][i]
                newtrainmatrix[j][3*i + 2][2][0] = 0
                
            for j in range(0, htest):
                newtestmatrix[j][3*i + 0][0][1] = testmatrix[j][rlist[0][0]]
                newtestmatrix[j][3*i + 0][1][1] = 0
                newtestmatrix[j][3*i + 0][2][1] = testmatrix[j][rlist[1][0]]
                newtestmatrix[j][3*i + 1][0][1] = 0
                newtestmatrix[j][3*i + 1][1][1] = testmatrix[j][i]
                newtestmatrix[j][3*i + 1][2][1] = testmatrix[j][rlist[2][0]]
                newtestmatrix[j][3*i + 2][0][1] = 0
                newtestmatrix[j][3*i + 2][1][1] = testmatrix[j][rlist[3][0]]
                newtestmatrix[j][3*i + 2][2][1] = 0
                
                newtestmatrix[j][3*i + 0][0][2] = rlist[0][1]
                newtestmatrix[j][3*i + 0][1][2] = 0
                newtestmatrix[j][3*i + 0][2][2] = rlist[1][1]
                newtestmatrix[j][3*i + 1][0][2] = 0
                newtestmatrix[j][3*i + 1][1][2] = 1
                newtestmatrix[j][3*i + 1][2][2] = rlist[2][1]
                newtestmatrix[j][3*i + 2][0][2] = 0
                newtestmatrix[j][3*i + 2][1][2] = rlist[3][1]
                newtestmatrix[j][3*i + 2][2][2] = 0
                
                newtestmatrix[j][3*i + 0][0][0] = testmatrix[j][i]
                newtestmatrix[j][3*i + 0][1][0] = 0
                newtestmatrix[j][3*i + 0][2][0] = testmatrix[j][i]
                newtestmatrix[j][3*i + 1][0][0] = 0
                newtestmatrix[j][3*i + 1][1][0] = testmatrix[j][i]
                newtestmatrix[j][3*i + 1][2][0] = testmatrix[j][i]
                newtestmatrix[j][3*i + 2][0][0] = 0
                newtestmatrix[j][3*i + 2][1][0] = testmatrix[j][i]
                newtestmatrix[j][3*i + 2][2][0] = 0
                
        return (newtrainmatrix, newtestmatrix)
    
    def getMatrix_CF32_8N8S8C(self, trainmatrix, testmatrix):
        (htrain, wtrain) = trainmatrix.shape
        (htest, wtest) = testmatrix.shape
        
        newtrainmatrix = numpy.zeros((htrain, wtrain * 3, 3, 3))
        newtestmatrix = numpy.zeros((htest, wtest * 3, 3, 3))
        
        for i in range(0, wtrain):
            corevector = trainmatrix[:, i]
            rlist = self.createPearsonRList(corevector, trainmatrix, i);
            
            for j in range(0, htrain):
                newtrainmatrix[j][3*i + 0][0][1] = trainmatrix[j][rlist[0][0]]
                newtrainmatrix[j][3*i + 0][1][1] = trainmatrix[j][rlist[1][0]]
                newtrainmatrix[j][3*i + 0][2][1] = trainmatrix[j][rlist[2][0]]
                newtrainmatrix[j][3*i + 1][0][1] = trainmatrix[j][rlist[3][0]]
                newtrainmatrix[j][3*i + 1][1][1] = trainmatrix[j][i]
                newtrainmatrix[j][3*i + 1][2][1] = trainmatrix[j][rlist[4][0]]
                newtrainmatrix[j][3*i + 2][0][1] = trainmatrix[j][rlist[5][0]]
                newtrainmatrix[j][3*i + 2][1][1] = trainmatrix[j][rlist[6][0]]
                newtrainmatrix[j][3*i + 2][2][1] = trainmatrix[j][rlist[7][0]]
                
                newtrainmatrix[j][3*i + 0][0][2] = rlist[0][1]
                newtrainmatrix[j][3*i + 0][1][2] = rlist[1][1]
                newtrainmatrix[j][3*i + 0][2][2] = rlist[2][1]
                newtrainmatrix[j][3*i + 1][0][2] = rlist[3][1]
                newtrainmatrix[j][3*i + 1][1][2] = 1
                newtrainmatrix[j][3*i + 1][2][2] = rlist[4][1]
                newtrainmatrix[j][3*i + 2][0][2] = rlist[5][1]
                newtrainmatrix[j][3*i + 2][1][2] = rlist[6][1]
                newtrainmatrix[j][3*i + 2][2][2] = rlist[7][1]
                
                newtrainmatrix[j][3*i + 0][0][0] = trainmatrix[j][i]
                newtrainmatrix[j][3*i + 0][1][0] = trainmatrix[j][i]
                newtrainmatrix[j][3*i + 0][2][0] = trainmatrix[j][i]
                newtrainmatrix[j][3*i + 1][0][0] = trainmatrix[j][i]
                newtrainmatrix[j][3*i + 1][1][0] = trainmatrix[j][i]
                newtrainmatrix[j][3*i + 1][2][0] = trainmatrix[j][i]
                newtrainmatrix[j][3*i + 2][0][0] = trainmatrix[j][i]
                newtrainmatrix[j][3*i + 2][1][0] = trainmatrix[j][i]
                newtrainmatrix[j][3*i + 2][2][0] = trainmatrix[j][i]
                
            for j in range(0, htest):
                newtestmatrix[j][3*i + 0][0][1] = testmatrix[j][rlist[0][0]]
                newtestmatrix[j][3*i + 0][1][1] = testmatrix[j][rlist[1][0]]
                newtestmatrix[j][3*i + 0][2][1] = testmatrix[j][rlist[2][0]]
                newtestmatrix[j][3*i + 1][0][1] = testmatrix[j][rlist[3][0]]
                newtestmatrix[j][3*i + 1][1][1] = testmatrix[j][i]
                newtestmatrix[j][3*i + 1][2][1] = testmatrix[j][rlist[4][0]]
                newtestmatrix[j][3*i + 2][0][1] = testmatrix[j][rlist[5][0]]
                newtestmatrix[j][3*i + 2][1][1] = testmatrix[j][rlist[6][0]]
                newtestmatrix[j][3*i + 2][2][1] = testmatrix[j][rlist[7][0]]
                
                newtestmatrix[j][3*i + 0][0][2] = rlist[0][1]
                newtestmatrix[j][3*i + 0][1][2] = rlist[1][1]
                newtestmatrix[j][3*i + 0][2][2] = rlist[2][1]
                newtestmatrix[j][3*i + 1][0][2] = rlist[3][1]
                newtestmatrix[j][3*i + 1][1][2] = 1
                newtestmatrix[j][3*i + 1][2][2] = rlist[4][1]
                newtestmatrix[j][3*i + 2][0][2] = rlist[5][1]
                newtestmatrix[j][3*i + 2][1][2] = rlist[6][1]
                newtestmatrix[j][3*i + 2][2][2] = rlist[7][1]
                
                newtestmatrix[j][3*i + 0][0][0] = testmatrix[j][i]
                newtestmatrix[j][3*i + 0][1][0] = testmatrix[j][i]
                newtestmatrix[j][3*i + 0][2][0] = testmatrix[j][i]
                newtestmatrix[j][3*i + 1][0][0] = testmatrix[j][i]
                newtestmatrix[j][3*i + 1][1][0] = testmatrix[j][i]
                newtestmatrix[j][3*i + 1][2][0] = testmatrix[j][i]
                newtestmatrix[j][3*i + 2][0][0] = testmatrix[j][i]
                newtestmatrix[j][3*i + 2][1][0] = testmatrix[j][i]
                newtestmatrix[j][3*i + 2][2][0] = testmatrix[j][i]
                
        return (newtrainmatrix, newtestmatrix)
        
        
    def getMatrix_C1D_4N4S (self, trainmatrix, testmatrix):
        '''
        For Conv1D with 4 neighbors and 4 similarity scores around the element
        {
        S4    N1    S1
        N4    El    N2
        S3    N3    S2
        }
        @param initmatrix: the initial matrix with all the runs
        @return: the final matrix   
        
        @todo: see if newtrainmatrix / newtestmatrix has to be (h, w*3, 3) or (h, 3, w*3)
        '''
        (htrain, wtrain) = trainmatrix.shape
        (htest, wtest) = testmatrix.shape
        
        newtrainmatrix = numpy.zeros((htrain, wtrain * 3, 3))
        newtestmatrix = numpy.zeros((htest, wtest * 3, 3))

        pearsonsum = 0

        for i in range(0, wtrain):
            corevector = trainmatrix[:, i]
            rlist = self.createPearsonRList(corevector, trainmatrix, i)
            
            for j in range(0, htrain):
                newtrainmatrix[j][3*i + 0][0] = trainmatrix[j][rlist[0][0]]
                newtrainmatrix[j][3*i + 0][1] = rlist[0][1]
                newtrainmatrix[j][3*i + 0][2] = trainmatrix[j][rlist[1][0]]
                newtrainmatrix[j][3*i + 1][0] = rlist[1][1]
                newtrainmatrix[j][3*i + 1][1] = trainmatrix[j][i]
                newtrainmatrix[j][3*i + 1][2] = trainmatrix[j][rlist[2][0]]
                newtrainmatrix[j][3*i + 2][0] = rlist[2][1]
                newtrainmatrix[j][3*i + 2][1] = trainmatrix[j][rlist[3][0]]
                newtrainmatrix[j][3*i + 2][2] = rlist[3][1]
                
            for j in range(0, htest):
                newtestmatrix[j][3*i + 0][0] = testmatrix[j][rlist[0][0]]
                newtestmatrix[j][3*i + 0][1] = rlist[0][1]
                newtestmatrix[j][3*i + 0][2] = testmatrix[j][rlist[1][0]]
                newtestmatrix[j][3*i + 1][0] = rlist[1][1]
                newtestmatrix[j][3*i + 1][1] = testmatrix[j][i]
                newtestmatrix[j][3*i + 1][2] = testmatrix[j][rlist[2][0]]
                newtestmatrix[j][3*i + 2][0] = rlist[2][1]
                newtestmatrix[j][3*i + 2][1] = testmatrix[j][rlist[3][0]]
                newtestmatrix[j][3*i + 2][2] = rlist[3][1]

            thispearsonsum = 0
            for k in range(0, wtrain - 1):
                thispearsonsum = thispearsonsum + rlist[k][1]
            pearsonsum = pearsonsum + (thispearsonsum / (wtrain - 1))

        avgpearson = pearsonsum / wtrain
        print("Average pearson is: ")
        print (avgpearson)
            #do the same for test
        
        return (newtrainmatrix, newtestmatrix)
    
    def createPearsonRList (self, corevector, trainmatrix, i):
        '''
        Create the Pearson R Correlation between a vector and a list
        '''
        (h, w) = trainmatrix.shape
        rlist = list()
        for j in range(0, w):
                if i == j:
                    continue
                else:
                    compvector = trainmatrix[:, j]
                    (rvalue, pvalue) = scipy.stats.stats.pearsonr(corevector, compvector)
                    
                    rlist.append((j, rvalue))
                    
        rlist = sorted(rlist, key=lambda tup: tup[1], reverse=True)
        
        return rlist

        
        
            
