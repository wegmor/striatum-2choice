import pandas as pd
import pathlib

def cachedDataFrame(filename):
    '''
    Decorator for marking an analysis function cachable. On first call to the function,
    the output (which must be a pandas DataFrame) is pickled and saved. On subsequent
    calls, the result is loaded from the pickle instead of recalculated. In the current
    implementation, no regard is taken to the arguments of the function when saving the
    pickle, so when the function is called with different arguments the file has to be
    manually deleted to trigger a recalculation.
    
    Arguments:
    filename -- The name of the file to save the pickle to. Should have the .pkl suffix.
    '''
    def decorator(fcn):
        def cachedDataFrameFunction(*args, cacheFolder=pathlib.Path("cache"), **kwargs):
            if not cacheFolder.is_dir():
                cacheFolder.mkdir()
            cachedDataPath = cacheFolder / filename
            if cachedDataPath.is_file():
                data = pd.read_pickle(cachedDataPath)
            else:
                data = fcn(*args, **kwargs)
                data.to_pickle(cachedDataPath)
            return data
        return cachedDataFrameFunction
    return decorator