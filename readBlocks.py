import pandas as pd
import numpy as np

import pyximport
pyximport.install()
import findPeaks

class Block:
    '''A block is one recording session.'''
    def __init__(self, hdfFile, path):
        self.hdfFile = hdfFile
        _, _, self.genotype, self.mouseNumber, self.date, self.recording, _ = path.split('/')
    
    def __str__(self):
        return self.genotype + "_" + self.mouseNumber + "_"+self.date

    def readROIs(self):
        '''Read the regions-of-interest (ROIs) for each neuron in this block. Contains the shape
        of the neurons.

        Returns:
        A Pandas panel with all the ROIs stacked on the item axis        
        '''
        return pd.read_hdf(self.hdfFile, self.getPath()+"neuronsROI")
    
    def readCaTraces(self, dropFirst=0):
        '''Read all calcium traces from this block.

        Arguments:
        dropFirst -- Number of seconds to drop from the beginning

        Returns:
        A Pandas dataframe with the calcium traces as columns
        '''
        path = self.getPath()+self.recording+"/"+'caTraces'
        caTraces = pd.read_hdf(self.hdfFile, path)
        return caTraces[dropFirst:]
    
    def readDeconvolvedTraces(self, dropFirst=0):
        '''Read all deconvolved traces from this block. Deconvolved traces are
        a guess of the "true" activity of the neuron.

        Arguments:
        dropFirst -- Number of seconds to drop from the beginning

        Returns:
        A Pandas dataframe with the calcium traces as columns
        '''
        path = self.getPath()+self.recording+"/"+'deconvolvedCaTraces'
        traces = pd.read_hdf(self.hdfFile, path)
        return traces[dropFirst:]
    
    def getPath(self):
        '''The path in the HDF file corresponding to this block.'''
        return "/data/"+self.genotype+"/"+self.mouseNumber+"/"+self.date+"/"
    
    def findPeaks(self, method="donahue", shape="wide"):
        '''Find "peaks" in the calcium traces.
        
        Arguments:
        method --- The method used to find the peaks
        shape --- Either "wide" or "long". Wide form is compatible with
                  caTraces and deconvolvedTraces, while long form is more
                  memory efficient for this sparse signal.
        '''
        return findPeaks.findPeaks(self.readCaTraces(), method=method, shape=shape)
    
    def readSensorValues(self, slim=True, onlyRecording=True, reindexFrameNo=True):
        '''Read the sensor values (LEDs and Beams) from this block.
        
        Arguments:
        slim -- If False, also include miscellanous parameters from the session
        onlyRecording -- If True, only return sensor values from when the Ca recording was on
        reindexFrameNo -- Reindex the frameNo variable so that 0 corresponds to the first row
                          of the calcium traces (which is the convention used by findPeaks)
                       
        Returns:
        A Pandas dataframe with the sensor values
        '''
        
        sensorValues = pd.read_hdf(self.hdfFile, self.getPath() + 'sensors')
        
        if slim:
            selectedColumns = ['time', 'trialNo', 'corrTrialNo', 'switchNo', 'rewardNo', 'rewardP',
                               'beamL', 'beamC', 'beamR', 'ledL', 'ledC', 'ledR', 'frameNo']
            sensorValues = sensorValues[selectedColumns]
        
        if onlyRecording:
            sensorValues = sensorValues[sensorValues.frameNo > 0].drop_duplicates("frameNo")
            #First and last frame are sometimes wrong (longer than 50 ms). In that case, throw them.
            if sensorValues.iloc[0].time < sensorValues.iloc[1].time-55:
                sensorValues = sensorValues[1:]
            if sensorValues.iloc[-2].time < sensorValues.iloc[-1].time-55:
                sensorValues = sensorValues[:-1]
        
        if reindexFrameNo:
            sensorValues.frameNo -= sensorValues.frameNo.iloc[0]
        return sensorValues

def findBlocks(hdfFile, onlyRecordedTrials=True, genotype=None, mouseNumber=None, date=None, recording=None):
    '''Generator of all experimental blocks stored in a HDF file.
    
    Arguments:
    hdfFile -- The filename of the HDF file
    genotype, mouseNumber, date -- Filters for selecting sessions. Specify as None (no filter, default),
                                   a specific value (string) or a list of values (any iterable)
    Example:
    >> for b in findBlocks('endoData.hdf', genotype=['a2a', 'oprm1']):
    >>     print b.genotype, b.mouseNumber, b.readROIs().shape
    '''
    store = pd.HDFStore(hdfFile)
    for k in store.keys():
        if k.endswith('/caTraces' if onlyRecordedTrials else '/sensors'):
            s = Block(store, k if onlyRecordedTrials else k[:-7]+'None/None')
            genOk = genotype is None or s.genotype == genotype or s.genotype in genotype
            numOk = mouseNumber is None or s.mouseNumber == str(mouseNumber) or s.mouseNumber in map(str, mouseNumber)
            datOk = date is None or s.date == date or s.date in date
            recOk = recording is None or s.recording == recording or s.recording in recording
            if all((genOk, numOk, datOk, recOk)):
                yield s
