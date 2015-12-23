#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import commanderline.commander_line as cl
# import scipy.signal as sig
import scipy.ndimage.filters as filters
import scipy.optimize
import sklearn.base
import sklearn.cross_validation

class LowPassFilterEstimator(sklearn.base.BaseEstimator):
    def __init__(self, filter_width=1):
        self.filter_width=filter_width
        
    def apply_filter(np_array, filter_width):
        # Apply boxcar filter 3x to approximate gaussian filter. 4x would be better.
        return filters.uniform_filter1d(
                filters.uniform_filter1d(
                 filters.uniform_filter1d(np_array, filter_width), 
                 filter_width), 
                filter_width)
        
    def fit(self, X, y):
        if self.filter_width % 2 == 0:
            print('Warning: in order to get a symmetric filter, filter_width should be a positive odd integer.')

        self.fitted_=pd.DataFrame(data=y, index=X)
        self.fitted_=self.fitted_.sort_index() # index may not be sorted :(

        self.fitted_['smooth']=LowPassFilterEstimator.apply_filter(self.fitted_[0].values, self.filter_width)

        return self
    
    def predict(self, X):
        i1=self.fitted_.index
        i2=i1.union(X) # this will sort index if possible

        # add new values to predict
        # interpolate inner values
        # fill leading NaN's
        # fill trailing NaN's
        self.predicted_=self.fitted_['smooth'].reindex(i2).interpolate(method='index').fillna(method='bfill') .fillna(method='ffill')

        return self.predicted_.loc[X]

    def score(self, X, y):
        pass
        
    # def predict_single(self, x_val):
    #     if x_val < 0:
    #         return self.resampled_smooth.iloc[0]
    #     elif x_val > len(self.resampled_smooth)-1:
    #         return self.resampled_smooth.iloc[len(self.resampled_smooth)-1]
    #     else:
    #         prev=self.resampled_smooth[self.resampled_smooth.index<=x_val].tail(1)
    #         next=self.resampled_smooth[self.resampled_smooth.index>=x_val].head(1)
    #         prev_x=prev.index[0]
    #         prev_y=prev.iloc[0]
    #         next_x=next.index[0]
    #         next_y=next.iloc[0]
    #         return prev_y+(x_val-prev_x)*(next_y-prev_y)/(next_x-prev_x)
                           
#     def score(self, X):
#         pass
        
    def get_params(self, deep=True):
        return {'filter_width': self.filter_width}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

def iteration(vector_in, filt_width, kf):
    score=0
    for train, test in kf:
        est=LowPassFilterEstimator(filt_width)
        d_train=vector_in.iloc[train]
        est.fit(d_train.index, d_train)
        # score=sum((est.predict(vector_in.iloc[test].index)-np.array(vector_in.iloc[test]))**2)
        score+=sum((est.predict(vector_in.iloc[test].index)-vector_in.iloc[test])**2)
    
    return score

def run(infile=None, outfile=None, cv_folds=10, basin_iters=100, sep=',', header=None):
    '''
    Cross-validated, approximately gaussian smoothing of equidistant time-series data

    Input is assumend to be a plain-text representation of a matrix with N rows and M columns. 
    Each row represents an (equidistant) timepoint, while each column represents a different time-series dataset.
    Input can be provided through stdin or filename via command-line argument (--infile).
    Input is assumed to be without a header (--header).
    Input column separator is ',' (--sep).
    Input can optionally be gzipped (compression is inferred automatically).
    '''
    # if no infile is provided, assume stdin; analogous for outfile
    if(infile is None):
        infile=sys.stdin
    if(outfile is None):
        outfile=sys.stdout

    # read input, use default separator, don't user header, and infer compression
    d_in=pd.read_csv(infile, sep=sep, header=header, compression='infer')

    # make sure all is float - avoid integer arithmetic in scipy.filters.uniform_filter1d (e.g., (1+2)/2=1)
    d_in=d_in.astype(float)

    # print(d_in)
    # print(len(d_in))

    kf = sklearn.cross_validation.KFold(len(d_in), n_folds=cv_folds, shuffle=False)

    d_out=pd.DataFrame()

    min_filters=[]
    num_increase=0
    for col in d_in:
        vector_in=d_in[col]

        min_filter=1
        min_score=float('inf')
        for filt_width in range(1, int(2*len(vector_in)*(1.0-1.0/cv_folds))-1, 2):
            score=iteration(vector_in, filt_width, kf)
            if score<=min_score:
                min_score=score
                min_filter=filt_width
            else:
                num_increase+=1
            # print(filt_width, score)
            if num_increase>=3:
              break

        d_out[col]=LowPassFilterEstimator.apply_filter(vector_in, min_filter)

    d_out.to_csv(outfile, sep=sep, header=header, index=False)

    # print(min_filter)
    # print(min_score)

# def run_min(infile=None, cv_folds=10, basin_iters=100, sep=',', header=None):
#     '''
#     Cross-validated, approximately gaussian smoothing of equidistant time-series data

#     Input is assumend to be a plain-text representation of a matrix with N rows and M columns. 
#     Each row represents an (equidistant) timepoint, while each column represents a different time-series dataset.
#     Input can be provided through stdin or filename via command-line argument (--infile).
#     Input is assumed to be without a header (--header).
#     Input column separator is ',' (--sep).
#     Input can optionally be gzipped (compression is inferred automatically).
#     '''
#     # if no infile is provided, assume stdin
#     if(infile is None):
#         infile=sys.stdin
#     # read input, use default separator, don't user header, and infer compression
#     d_in=pd.read_csv(infile, sep=sep, header=header, compression='infer')

#     # make sure all is float - avoid integer arithmetic in scipy.filters.uniform_filter1d (e.g., (1+2)/2=1)
#     d_in=d_in.astype(float)

#     kf = sklearn.cross_validation.KFold(len(d_in), n_folds=cv_folds, shuffle=False)

#     for col in d_in:
#         vector_in=d_in[col]

#         result=scipy.optimize.minimize(lambda x: iteration(vector_in, int(x), kf), x0=[1], bounds=[(1,int(2*len(vector_in)*(1.0-1.0/cv_folds))-1)])
#         print(result)


cl.commander_line(run, print_argv_to_output=False, print_done=False) if __name__ == '__main__' else None
