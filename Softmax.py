import numpy as np
import math

def fn( x ):
    print( x/x.sum() )

def softmax( x ):
    e = np.exp( x )
    print( e )
    print("=============")
    print( e/e.sum() )

arr = np.array( [2.0,1.0, 0.1] )
# fn( arr )
softmax( arr )