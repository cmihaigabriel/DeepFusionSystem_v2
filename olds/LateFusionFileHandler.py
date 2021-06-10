'''
Created on Jan 25, 2019

@author: cmihaigabriel@gmail.com
@version: v2.1

@change: v1.0 used in 2019 SI IJCV
@change: v2.0 used in 2020 ICMR
@change: v2.1 used in 2020 ECCV
'''

import pathlib
from os import listdir
from os.path import isfile, join
import csv
import os
import numpy
from builtins import str
import subprocess
import re
from scipy.stats.mstats import pearsonr
from sklearn.metrics import f1_score

class FileHandler ():
    '''
    Handles operations for the run files:
    - loading all run files in a matrix
    - sorting the matrix according to METRIC scores of each run
    - saving a final file with the final prediction values, to be used after running the fusion algoritm
    '''
    
    def __init__ (self):
        self.loadfolder = ""
        self.savefolder = ""
        self.savenames = ""
        self.gtdata = ""
        self.trecpath = ""
        self.expectedsamples = 0
        self.expectedruns = 0
        self.partialgtfilename = ''
        self.metrictype = 0
        self.competitioncode = 0
        
        self.warnings = 0
        self.warntext = ' '
        self.gtfilenames = []
        self.gtscores = []
        self.gtpredictions = []
        
        self.runcollection = []
        self.runcollection_sorted = []
        
        self.metric_gt_ver = 1
        
    def setup (self, loadf, savef, saven, gt, trec, samples, runs, metrictype, competitioncode):
        '''
        sets up the folders and names
        @param loadf: string : the folder that contains all the runs
        @param savef: string : the folder where all the data will be saved
        @param saven: string : the filename root for the saved files
        @param gt: string : the path to gt data
        @param trecpath: string : the path to trec_eval
        @param samples: int : the number of samples in each runfile - WILL CREATE A WARNING IF THE PROCESSED NUMBER IS LOWER / HIGHER THAN EXPECTED
        @param runs: int : the number of runs in this task - WILL CREATE A WARNING IF THE PROCESSED NUMBER IS LOWER / HIGHER THAN EXPECTED
        @param metrictype: int: the type of calculated metric:
                                - 0 MAP - using competition tool
                                - 1 MAP@10 - using competition tool
                                - 2 F1 - quite simple therefore used the direct formula (tested and got the same results as the comp tool)
                                - 3 MSE and PCC - simple direct formula (tested...)
                                - 4 IoU - simple direct formula (tested...)
        @param competitioncode : int : the code of the competition:
                                - 0 INT2017.Video
                                - 1 INT2017.Image
                                - 2 VSD2015
                                - 3 MEDCapt2019
                                - 4 EMOAroVal2018.Arousal
                                - 5 EMOAroVal2018.Valence
                                - 6 EMOFear2018
        '''
        self.loadfolder = loadf
        self.savefolder = savef
        self.savenames = saven
        self.gtdata = gt
        self.trecpath = trec
        self.expectedsamples = samples
        self.expectedruns = runs
        self.metrictype = metrictype
        self.competitioncode = competitioncode
        
    '''
    -------------
    File IO handling functions
    -------------
    '''
    def loadFiles (self):
        '''
        loads the files in an object
        @param task: int: defines the task
                        : implemented 1 = MediaEval Interestingness
        '''
        self.warnings = 0
        self.gtfilenames = self.process_GtFilenames(self.gtdata)

        #
        # INT2017.Video or INT2017.Image
        #
        if self.competitioncode == 1 or self.competitioncode == 0 :
            allfiles = [f for f in sorted(listdir(self.loadfolder)) if isfile(join(self.loadfolder, f))]
            for filename in allfiles:
                f = join(self.loadfolder, filename)
                froot = filename.split(os.extsep)[0]
                fext = pathlib.Path(filename).suffix
                
                if (fext == '.log'):
                    mapmetric = self.process_LogScore(f)
                    #print('Processing MAPS16 @ ... ' + f + ' with MAP = ' + str(mapmetric))
                    self.updateRunCollection(froot, mapmetric, None, None, None, None)
                
                if (fext == '.txt') :
                    print(f)
                    scores = self.process_CsvScores(f)
                    #print('Processed Scores @ ... ' + f)
                    scoresNormed, runmin, runmax = self.normalizeScores(0, 1, scores)
                    self.updateRunCollection(froot, None, scores, runmin, runmax, scoresNormed)
                    
        #
        # VSD2015
        #
        if self.competitioncode == 2 :
            allfiles = [f for f in sorted(listdir(self.loadfolder)) if isfile(join(self.loadfolder, f))]
            for filename in allfiles:
                f = join(self.loadfolder, filename)
                froot = filename.split(os.extsep)[0]
                fext = pathlib.Path(filename).suffix
                
                if (fext == '.log'):
                    mapmetric = self.process_LogScore(f)
                    #print('Processing MAPS16 @ ... ' + f + ' with MAP = ' + str(mapmetric))
                    self.updateRunCollection(froot, mapmetric, None, None, None, None)
                
                if (fext == '.txt') :
                    print(f)
                    scores = self.process_CsvScores(f)
                    #print('Processed Scores @ ... ' + f)
                    scoresNormed, runmin, runmax = self.normalizeScores(-1, 1, scores)
                    self.updateRunCollection(froot, None, scores, runmin, runmax, scoresNormed)

        #
        # MEDCap2019
        #
        if self.competitioncode == 3:
            allfiles = [f for f in sorted(listdir(self.loadfolder)) if isfile(join(self.loadfolder, f))]
            for filename in allfiles:
                f = join(self.loadfolder, filename)
                froot = filename.split(os.extsep)[0]
                fext = pathlib.Path(filename).suffix

                if fext == '.csvmasklog':
                    mapmetric = self.process_LogScore(f)
                    print('processing csvmasklog')
                    self.updateRunCollection(froot, mapmetric, None, None, None, None)
                if fext == '.csvmask':
                    print(f)
                    scores = self.process_CsvScores(f)
                    #no normalization for these tasks
                    #scoresNormed, runmin, runmax = self.normalizeScores(0, 1, scores)
                    runmin = self.findMinimumList(scores)
                    runmax = self.findMaximumList(scores)
                    scoresNormed = scores
                    self.updateRunCollection(froot, None, scores, runmin, runmax, scoresNormed)

        #
        # EMOAroval2018.Arousal and EMOAroval2018.Valence
        #
        if self.competitioncode == 4 or self.competitioncode == 5:
            allfiles = [f for f in sorted(listdir(self.loadfolder)) if isfile(join(self.loadfolder, f))]
            for filename in allfiles:
                f = join(self.loadfolder, filename)
                froot = filename.split(os.extsep)[0]
                fext = pathlib.Path(filename).suffix

                if (fext == '.alog' and self.competitioncode == 4):
                    mapmetric = self.process_LogScore(f)
                    print('processing alog')
                    self.updateRunCollection(froot, mapmetric, None, None, None, None)

                if (fext == '.vlog' and self.competitioncode == 5):
                    mapmetric = self.process_LogScore(f)
                    print('processing vlog')
                    self.updateRunCollection(froot, mapmetric, None, None, None, None)

                if (fext == '.txtnew') :
                    print(f)
                    scores = self.process_CsvScores(f)
                    #no normalization for these tasks
                    #scoresNormed, runmin, runmax = self.normalizeScores(0, 1, scores)
                    runmin = self.findMinimumList(scores)
                    runmax = self.findMaximumList(scores)
                    scoresNormed = scores
                    self.updateRunCollection(froot, None, scores, runmin, runmax, scoresNormed)

        #
        # EMOFear2018
        #
        if self.competitioncode == 6:
            allfiles = [f for f in sorted(listdir(self.loadfolder)) if isfile(join(self.loadfolder, f))]
            for filename in allfiles:
                f = join(self.loadfolder, filename)
                froot = filename.split(os.extsep)[0]
                fext = pathlib.Path(filename).suffix

                if (fext == '.log'):
                    mapmetric = self.process_LogScore(f)
                    print('processing log')
                    self.updateRunCollection(froot, mapmetric, None, None, None, None)

                if (fext == '.txt2'):
                    print(f)
                    scores = self.process_CsvScores(f)
                    runmin = 0
                    runmax = 1
                    scoresNormed = scores
                    self.updateRunCollection(froot, None, scores, runmin, runmax, scoresNormed)
         
        #
        # End of tasks
        #
        print('Finished processing files')
        
        if self.competitioncode in [4,5] :
            self.runcollection_sorted = self.sortRunCollection(self.runcollection, 0)
        else:
            self.runcollection_sorted = self.sortRunCollection(self.runcollection, 1)
           
        if (self.warnings > 0):
            print(self.warntext)
            print('WARNING!!! ... at least one warning was generated. Check the logs or proceed with great care')
        self.warnings = 0
        self.warntext = ' '
        
    def saveFiles (self, scores, fnamesuf, validation_mask = None):
        '''
        saves the final results
        @param scores: the score list
        @param fnamesuf: the suffix of the name - without extension
        @param validation_mask: mask to be used in case of kfolds
        @return: metric value
        '''
        #
        # INT2017.Video or INT2017.Image
        #
        if self.competitioncode == 1 or self.competitioncode == 0 :
            savefname = self.savefolder + self.savenames + fnamesuf + '.txt.trec'
            f = open(savefname, 'w')
            if validation_mask is not None:
                self.savegtpartial(validation_mask)
                self.metric_gt_ver = 2
                for i in range(0, scores.shape[0]):
                    idx = validation_mask[i]
                    sample = self.gtfilenames[idx]
                    score = scores[i]
                    movname, shotname = sample.split(',')
                    intrunname = self.savenames + fnamesuf + '.txt'
                    f.write(movname + ' 0 ' + shotname + ' 0 ' + str(score) + ' ' + intrunname + '\n')
            if validation_mask is None:
                self.metric_gt_ver = 1
                for i in range(0, scores.shape[0]):
                    sample = self.gtfilenames[i]
                    score = scores[i]
                    movname, shotname = sample.split(',')
                    intrunname = self.savenames + fnamesuf + '.txt'
                    f.write(movname + ' 0 ' + shotname + ' 0 ' + str(score) + ' ' + intrunname + '\n')
            f.close()
            
        #
        # VSD2015
        #
        if self.competitioncode == 2 :
            savefname = self.savefolder + self.savenames + fnamesuf + '.txt.trec'
            f = open(savefname, 'w')
            if validation_mask is not None:
                self.savegtpartial(validation_mask)
                self.metric_gt_ver = 2
                for i in range(0, scores.shape[0]):
                    idx = validation_mask[i]
                    sample = self.gtfilenames[idx]
                    score = scores[i]
                    intrunname = self.savenames + fnamesuf + '.txt'
                    f.write('violence 0 ' + sample + ' 0 ' + str(score) + ' ' + intrunname + '\n')
            if validation_mask is None:
                self.metric_gt_ver = 1
                for i in range(0, scores.shape[0]):
                    sample = self.gtfilenames[i]
                    score = scores[i]
                    intrunname = self.savenames + fnamesuf + '.txt'
                    f.write('violence 0 ' + sample + ' 0 ' + str(score) + ' ' + intrunname + '\n')
            f.close()           
            
        metric = self.getMetric(savefname)
        return metric
    
    def saveSimpleCsv(self, filepath, filecontent):
        '''
        saves a csv file
        @param filepath: the file path - without the name - name will be ::saven::.csv from setup
        @param filecontent: the file content
        '''
        f = open(filepath + self.savenames + '.csv', 'w')
        f.write(filecontent)
        f.close()

    def savegtpartial(self, validation_mask):
        '''
        saves a partial version of the GT file
        can be used for measuring MAP on k-folds
        @param validation_mask: list of int: represent idx of each sample
        '''
        #
        # INT2017.Video or INT2017.Image
        #
        if self.competitioncode == 1 or self.competitioncode == 0 :
            savefname = self.savefolder + 'temp_gt_kfold' + '.qrels'
            f = open(savefname, 'w')
            for i in range(0, len(validation_mask)):
                idx = validation_mask[i]
                sample = self.gtfilenames[idx]
                gtclassification = self.gtpredictions[idx]            
                movname, shotname = sample.split(',')
                f.write(movname + ' 0 ' + shotname + ' ' + str(gtclassification) + '\n')
            f.close()
            self.partialgtfilename = savefname
            
        #
        # VSD2015
        #
        if self.competitioncode == 2 :
            savefname = self.savefolder + 'temp_gt_kfold' + '.qrel'
            f = open(savefname, 'w')
            for i in range(0, len(validation_mask)):
                idx = validation_mask[i]
                sample = self.gtfilenames[idx]
                gtclassification = self.gtpredictions[idx]            
                f.write('violence 0 ' + sample + ' ' + str(gtclassification) + '\n')
            f.close()
            self.partialgtfilename = savefname


    def getMetric (self, fname):
        '''
        saves the final results
        @param fname: filename
        @retrun: metric value
        '''
        
        mapret = 0.0
        
        #
        # INT2017.Video or INT2017.Image
        #
        if self.competitioncode == 1 or self.competitioncode == 0 :
            cutoffvalue = ""
            if self.metrictype == 1:
                cutoffvalue = " -M10"
            command = self.trecpath + cutoffvalue + ' \"' + self.gtdata + '\" \"' + fname + '\"'
            if self.metric_gt_ver == 2: #using an incomplete GT file
                command = self.trecpath + cutoffvalue + ' \"' + self.partialgtfilename + '\" \"' + fname + '\"'
            #print(command)
            output = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True).stdout
            for line in output:
                linestr = str(line)
                idxmap = linestr.find('map')
                if idxmap > -1:
                    mapret = re.findall("\d+\.\d+", linestr)[0]
                    
        #
        # VSD2015
        #
        if self.competitioncode == 2 : 
            cutoffvalue = ""
            if self.metrictype == 1:
                cutoffvalue = " -M10"
            command = self.trecpath + cutoffvalue + ' -a \"' + self.gtdata + '\" \"' + fname + '\"'
            if self.metric_gt_ver == 2: #using an incomplete GT file
                command = self.trecpath + cutoffvalue + ' -a \"' + self.partialgtfilename + '\" \"' + fname + '\"'
            #print(command)
            output = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True).stdout
            for line in output:
                linestr = str(line)
                idxmap = linestr.find('map')
                if idxmap > -1:
                    mapret = re.findall("\d+\.\d+", linestr)[0]          
        return mapret


    '''
    -------------
    Individual filetype processors
    -------------
    '''
    def process_CsvScores (self, filename):
        '''
        processes a CSV-ish score file for a certain RUN
        @param filename: str: the file
        @return: scores: list of floats: the loaded score values
        '''
        line_counter = 0
        scores = []
        score_dict = {}
        
        #
        # INT2017.Video or INT2017.Image
        #
        if self.competitioncode == 1 or self.competitioncode == 0 :
            with open(filename, mode='r') as f:
                csv_reader = csv.DictReader(f, fieldnames=('mov', 'shot', 'class', 'score'))
                for row in csv_reader :
                    #line_counter += 1
                    samplename = row['mov'] + ',' + row['shot']
                    score = float(row['score'])
                    score_dict[samplename] = score
                    
            minscore = self.findMinimumDict(score_dict)
                    
            for i in range(len(self.gtfilenames)):
                samplesearch = self.gtfilenames[i]
                
                if samplesearch in score_dict:
                    scores.append(score_dict[samplesearch])
                else:
                    scores.append(minscore)
                    
        #
        # VSD2015
        #
        if self.competitioncode == 2 :
            with open(filename, mode='r') as f:
                csv_reader = csv.DictReader(f, fieldnames=('shot', 'score', 'class'), delimiter = ' ')
                for row in csv_reader :
                    #line_counter += 1
                    samplename = row['shot']
                    score = float(row['score'])
                    score_dict[samplename] = score
                    
            minscore = self.findMinimumDict(score_dict)
                    
            for i in range(len(self.gtfilenames)):
                samplesearch = self.gtfilenames[i]
                
                if samplesearch in score_dict:
                    scores.append(score_dict[samplesearch])
                else:
                    scores.append(minscore)

        #
        # MEDCap2019 
        #
        if self.competitioncode == 3:
            with open(filename, mode='r') as f:
                csv_reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
                for row in csv_reader:
                    samplename = row[0]
                    score = list(map(int, row[1:]))
                    score_dict[samplename] = score

            minscore = 0

            for i in range(len(self.gtfilenames)):
                samplesearch = self.gtfilenames[i]
                
                if samplesearch in score_dict:
                    scores.append(score_dict[samplesearch])
                else:
                    scores.append([0] * len(self.gtfilenames[i]))

        #
        # EMOAroval2018.Arousal and EMOAroval2018.Valence
        #
        if self.competitioncode == 4 or self.competitioncode == 5:
            with open(filename, mode='r') as f:
                csv_reader = csv.DictReader(f, fieldnames=('movie', 'valence', 'arousal'), delimiter = ',')
                for row in csv_reader:
                    samplename = row['movie']
                    score = 0.0
                    if self.competitioncode == 4:
                        score = float(row['arousal'])
                    else:
                        score = float(row['valence'])
                    score_dict[samplename] = score

                #this rewrite will create a better type of scorefile that is easier to handle
                #remember to comment it out along with junks
                #self.junk_SaveNewtypeCsv(filename,junkname,junkval,junkaro)

            minscore = self.findMinimumDict(score_dict)

            for i in range(len(self.gtfilenames)):
                samplesearch = self.gtfilenames[i]
                
                if samplesearch in score_dict:
                    scores.append(score_dict[samplesearch])
                else:
                    scores.append(0.0)

        #
        # EMOFear2018
        #
        if self.competitioncode == 6:
            with open(filename, mode='r') as f:
                csv_reader = csv.DictReader(f, fieldnames=('samplename', 'score'), delimiter = ',')
                for row in csv_reader:
                    samplename = row['samplename']
                    score = int(row['score'])
                    score_dict[samplename] = score

            minscore = 0

            for i in range(len(self.gtfilenames)):
                samplesearch = self.gtfilenames[i]

                if samplesearch in score_dict:
                    scores.append(score_dict[samplesearch])
                else:
                    scores.append(0.0)

        return scores
    
    def process_LogScore(self, filename):
        '''
        process the LOG file, searching for METRIC for a certain RUN
        @param filename: str: the file
        @return: mapmetric: float: the corresponding METRIC score
        '''
        mapmetric = 0.0

        #
        # INT2017.Video, INT2017.Image and VSD2015
        #
        if self.competitioncode in [0,1,2]:
            with open(filename, mode='r') as f:
                csv_reader = csv.DictReader(f, fieldnames=('metric', 'video', 'score'), delimiter = '\t')
                for row in csv_reader :
                    if ( (row['metric']).strip() == 'map' ) and ( (row['video']).strip() == 'all' ) :
                        mapmetric = float( (row['score']).strip() )

        #
        # EMOAroval2018.Arousal and EMOAroval2018.Valence and EMOFear.2018
        #
        if self.competitioncode in [3,4,5,6]:
            with open(filename, mode='r') as f:
                csv_reader = csv.DictReader(f, fieldnames=('score', 'empty'), delimiter=' ')
                for row in csv_reader:
                    mapmetric = float((row['score']))
                    
        return mapmetric
    
    def process_GtFilenames(self, filename):
        '''
        loads all the filenames from the GT file
        will be used to allign all the run files to the same order
        this is mandatory as it will make sure that all samples have the runscores, therefore taking care of
                incomplete runfiles

        @param filename: str: the file
        @return: filenames: list of str: list of the loaded (run)names
        '''
        line_counter = 0
        filenames = []
        self.gtscores = []
        self.gtpredictions = []
        self.ioutypemaxlen = []
        
        #
        # INT2017.Video and INT2017.Image
        #
        if self.competitioncode == 1 or self.competitioncode == 0 :
            with open(filename, mode='r') as f:
                csv_reader = csv.DictReader(f, fieldnames = ('mov', 'unknwn', 'shot', 'class'), delimiter = ' ')
                for row in csv_reader:
                    line_counter += 1
                    filenames.append(row['mov'] + ',' + row['shot'])
                    self.gtpredictions.append(int(row['class']))
                    
            if (self.expectedsamples != line_counter) or (len(filenames) != self.expectedsamples):
                self.warnings += 1
                self.warntext += 'WARNING!!! ... missmatch between the number of loaded and expected Ground Truth items for GT file   ' + filename
    
            filenamescores = filename + '.txt'
            with open(filenamescores, mode='r') as f:
                csv_reader = csv.DictReader(f, fieldnames = ('mov', 'shot', 'class', 'score', 'rank'), delimiter = ',')
                for row in csv_reader:
                    self.gtscores.append(float(row['score']))
                self.gtscores, minval, maxval = self.normalizeScores(0, 1, self.gtscores)
                
        #
        # VSD2015
        #
        if self.competitioncode == 2 :
            with open(filename, mode='r') as f:
                csv_reader = csv.DictReader(f, fieldnames = ('unknwn1', 'unknwn2', 'shot', 'class'), delimiter = ' ')
                for row in csv_reader:
                    line_counter += 1
                    filenames.append(row['shot'])
                    self.gtpredictions.append(int(row['class']))
                    
            if (self.expectedsamples != line_counter) or (len(filenames) != self.expectedsamples):
                self.warnings += 1
                self.warntext += 'WARNING!!! ... missmatch between the number of loaded and expected Ground Truth items for GT file   ' + filename

        #
        # MEDCap2019
        #
        if self.competitioncode == 3:
            with open(filename, mode='r') as f:
                csv_reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
                for row in csv_reader:
                    line_counter += 1
                    filenames.append(row[0])
                    self.gtpredictions.append(list(map(int, row[1:len(row)-1])))
                    
            if (self.expectedsamples != line_counter) or (len(filenames) != self.expectedsamples):
                self.warnings += 1
                self.warntext += 'WARNING!!! ... missmatch between the number of loaded and expected Ground Truth items for GT file   ' + filename

        #
        # EMOAroval2018.Arousal and EMOAroval2018.Valence
        #
        if self.competitioncode == 4 or self.competitioncode == 5:
            with open(filename, mode='r') as f:
                csv_reader = csv.DictReader(f, fieldnames = ('samplename', 'valence', 'arousal'), delimiter = ',')
                for row in csv_reader:
                    line_counter += 1
                    filenames.append(row['samplename'])
                    if self.competitioncode == 4:
                        self.gtpredictions.append(float(row['arousal']))
                        self.gtscores.append(float(row['arousal']))
                    if self.competitioncode == 5:
                        self.gtpredictions.append(float(row['valence']))
                        self.gtscores.append(float(row['valence']))

            if (self.expectedsamples != line_counter) or (len(filenames) != self.expectedsamples):
                self.warnings += 1
                self.warntext += 'WARNING!!! ... missmatch between the number of loaded and expected Ground Truth items for GT file   ' + filename

        #
        # EMOFear2018
        #
        if self.competitioncode == 6:
            with open(filename, mode='r') as f:
                csv_reader = csv.DictReader(f, fieldnames=('samplename', 'score'), delimiter=',')
                for row in csv_reader:
                    line_counter += 1
                    filenames.append(row['samplename'])
                    self.gtpredictions.append(int(row['score']))
                    self.gtscores.append(int(row['score']))

            if (self.expectedsamples != line_counter) or (len(filenames) != self.expectedsamples):
                self.warnings += 1
                self.warntext += 'WARNING!!! ... missmatch between the number of loaded and expected Ground Truth items for GT file   ' + filename

        return filenames
    
    '''
    -------------
    Run Collection & Vector handling functions
    -------------
    '''
    def updateRunCollection (self, name, metric, scores, minval, maxval, scoresnormed):
        '''
        Updates (add / edit) a list with all the runs 
        of type RunDetails
        @param name: str: name of the run
        @param metric: float: value of the metric
        @param scores: list of float: list of the individual sample scores
        @param minval: float: minimum score value
        @param maxval: float: maximum score value
        @param scoresnormed: list of float: list of the normed sample scores
        '''
        if name is None:
            return
        
        runfound = False
        for i in range(len(self.runcollection)):
            if self.runcollection[i].runname == name:
                runfound = True
                if metric is not None:
                    self.runcollection[i].runmetric = metric
                if scores is not None:
                    self.runcollection[i].runscores = scores
                if min is not None:
                    self.runcollection[i].runscore_min = minval
                if max is not None:
                    self.runcollection[i].runscore_max = maxval
                if scoresnormed is not None:
                    self.runcollection[i].runscores_normed = scoresnormed
                    
        if runfound == False:
            newrun = RunDetails()
            newrun.runname = name
            if metric is not None:
                newrun.runmetric = metric
            if scores is not None:
                newrun.runscores = scores
            if min is not None:
                newrun.runscore_min = minval
            if max is not None:
                newrun.runscore_max = maxval
            if scoresnormed is not None:
                newrun.runscores_normed = scoresnormed
            self.runcollection.append(newrun)
            
    def sortRunCollection(self, collection, sorttype):
        '''
        Sorts the run collection and returns the sorted collection
        @param collection: the collection
        @param sorttype: the type of sorting
                        : implemented 1 = descending sorting
        @return: the sorted run collection
        '''
        newlist = []
        if sorttype == 1:
            newlist = sorted(collection, key = lambda x: x.runmetric, reverse=True)
        elif sorttype == 0:
            newlist = sorted(collection, key = lambda x: x.runmetric, reverse=False)
            
        return newlist    

    '''
    -------------
    Statistical and Math functions for calculating some helpful stuff
    -------------
    '''
    def normalizeScores(self, absmin, absmax, scores):
        '''
        Performs normalization in the absmin-absmax interval for a given vector
        @param absmin: minimum possible value
        @param absmax: maximum possible value
        @param scores: list of floats : the individual sample scores
        @return: scoresNormed: list of floats: the individual sample scores, normalized
        @return: runmin: the minimum original score value
        @return: runmax: the maximum original score value
        '''
        scoresNormed = []
        
        runmin = self.findMinimumList(scores)
        runmax = self.findMaximumList(scores)
        for i in range(len(scores)):
            n = float((scores[i]) - runmin)/float(runmax - runmin)
            scoresNormed.append(n)
        
        return scoresNormed, runmin, runmax
    
    def findMinimumList(self, scores):
        minimum = None
        for i in range(len(scores)):
            if minimum == None:
                minimum = scores[i]
            else:
                if minimum > scores[i]:
                    minimum = scores[i]
        
        return minimum
    
    def findMaximumList(self, scores):
        maximum = None
        for i in range(len(scores)):
            if maximum == None:
                maximum = scores[i]
            else:
                if maximum < scores[i]:
                    maximum = scores[i]
        
        return maximum
    
    def findMinimumDict(self, dicti):
        '''
        finds Minimum Value from a dict
        @return: minimum: float min value
        '''
        minimum = None
        
        for key in dicti:
            if minimum == None:
                minimum = dicti[key]
            else:
                if minimum > dicti[key]:
                    minimum = dicti[key]
                    
        return minimum
    
    '''
    -------------
    Values getter functions
    -------------
    '''
    def getScoresNormalized(self):
        scores = numpy.zeros((len(self.runcollection), len(self.runcollection[0].runscores_normed)))
        for i in range(len(self.runcollection)) :
            scores[i] = self.runcollection[i].runscores_normed
        
        fscores = scores.transpose()
        return fscores
    
    def getScoresNormalizedSorted(self):
        if self.competitioncode == 3:
            scores = numpy.zeros((len(self.runcollection_sorted), len(self.runcollection_sorted[0].runscores_normed), len(self.runcollection_sorted[0].runscores_normed[0])))
            #print(str(scores.shape))
        else:
            scores = numpy.zeros((len(self.runcollection_sorted), len(self.runcollection_sorted[0].runscores_normed)))
            #print(str(scores.shape))
        for i in range(len(self.runcollection_sorted)) :
            scores[i] = self.runcollection_sorted[i].runscores_normed
        
        if self.competitioncode == 3:
            fscores = scores.transpose(1, 2, 0)
            #print(str(fscores.shape))
        else:
            fscores = scores.transpose()
            #print(str(fscores.shape))
        return fscores

    def getGtScores(self):
        scores = numpy.zeros(len(self.gtscores))
        for i in range(len(self.gtscores)):
            scores[i] = self.gtscores[i]

        fscores = scores.transpose()
        return fscores
    
    def getGtPredictions(self):
        if self.competitioncode == 3:
            preds = numpy.zeros((len(self.gtpredictions), len(self.gtpredictions[0])))
        else:
            preds = numpy.zeros(len(self.gtpredictions))
        for i in range(len(self.gtpredictions)):
            preds[i] = self.gtpredictions[i]

        #print(str(preds.shape))
            
        return preds
    '''
    -------------
    Calculations for some basic metrics
    -------------
    '''
    def calcPearsonCC(self, pred, gt):
        '''
        Calculates Pearson's Correlation Coefficient
        '''
        pcc = pearsonr(pred,gt)[0]
        return pcc

    def calcF1(self, pred, gt):
        count = len(gt)
        f1sum = 0.0
        f1score = 0.0

        for i in range(len(gt)):
            f1sample = f1_score(gt[i], pred[i], average='binary')
            f1sum += f1sample

        f1score = f1sum / count
        return f1score

    def calcIoU(self, pred, gt):
        '''
        Calculate an overall IoU
        '''

        sampleI = numpy.array(pred).astype(int) & numpy.array(gt).astype(int)
        sampleU = numpy.array(pred).astype(int) | numpy.array(gt).astype(int)

        finiou = float(sampleI.sum()) / float(sampleU.sum())

        return finiou

    def calcIoUPerSample(self, pred, gt, mask):
        '''
        Calculate an overall IoU on a per sample base
        '''
        finiou = 0.0

        filenames = list(self.gtfilenames[i] for i in mask)

        listmovies = []
        if self.competitioncode == 6:
            listmovies = ['MEDIAEVAL18_54', 'MEDIAEVAL18_55', 'MEDIAEVAL18_56', 'MEDIAEVAL18_57', 'MEDIAEVAL18_58', 'MEDIAEVAL18_59', 'MEDIAEVAL18_60',
                'MEDIAEVAL18_61', 'MEDIAEVAL18_62', 'MEDIAEVAL18_63', 'MEDIAEVAL18_64', 'MEDIAEVAL18_65', 'MEDIAEVAL18_66']

        iousamples = []
        for mov in listmovies:
            index = [i for i, s in enumerate(filenames) if mov in s]
            if len(index) > 0:
                ypredsample = [pred[j] for j in index]
                ygtsample = [gt[j] for j in index]
                
                sampleI = numpy.array(ypredsample).astype(int) & numpy.array(ygtsample).astype(int)
                sampleU = numpy.array(ypredsample).astype(int) | numpy.array(ygtsample).astype(int)

                if (sampleU.sum() > 0):
                    iousamples.append(float(sampleI.sum()) / float(sampleU.sum()))
                else:
                    iousamples.append(1.0)

        finiou = sum(iousamples) / len(iousamples)

        return finiou

    def calc1DThresh(self, pred):
        '''
        Calculate the threshold for binary classification in an 1D y scenario
        '''
        labels = list(self.getGtPredictions())
        ones = labels.count(1.0)
        tots = len(labels)
        perc = float(ones)/float(tots)
        pozsamples = int(float(len(pred))*float(perc))

        pred_s = numpy.sort(pred,axis=None)[::-1]
        return pred_s[pozsamples]

    def calc2DThresh(self, pred):
        labels = list(self.getGtPredictions().transpose())
        pred_t = pred.transpose()

        ones = []
        for i in range(len(labels)):
            one = numpy.count_nonzero(labels[i] == 1.0)
            ones.append(one)

        tots = len(labels[0])
        percs = []
        for i in range(len(ones)):
            perc = float(ones[i])/float(tots)
            percs.append(perc)

        preds_s = []
        for i in range(len(percs)):   
            pozsamples = int(float(tots) * float(perc))

            pred_s = numpy.sort(pred_t[i],axis=None)[::-1]
            preds_s.append(pred_s[pozsamples])

        return preds_s

    '''
    -------------
    These are some junk functions that help with some pre-tasks (i.e. rewriting score files to be easier to handle)
    -------------
    '''
    def junk_SaveNewtypeCsv(self, filename, junknames, junkval, junkaro):
        '''
        Rewriting the EMOAroval gt and pred files
        '''
        newfilename = filename + "new"
        with open(newfilename, mode='w') as f:
            for i in range(len(junknames)):
                f.write(junknames[i] + "," + str(junkval[i]) + "," + str(junkaro[i]) + "\n")
            f.close()
        
        
class RunDetails:
    '''
    Collection handling all the runs, in a single object
    @var runname: str: the corresponding name of the run / runfiles
    @var runmetric: float: the metric value obtained by the run
    @var runscores: list of float: the original recorded scores
    @var runscore_min: float: the original minimum score
    @var runscore_max: float: the original maximum score
    @var runscores_normed: list of float: the normalized recorded scores
    '''
    
    def __init__ (self):
        self.runname = ''
        self.runmetric = 0.0
        self.runscores = []
        self.runscore_min = 0.0
        self.runscore_max = 0.0
        self.runscores_normed = []
        
