import numpy as np
import re
import sys

def str2float( string2parse : str ):
    """
        Return A list of float that is parsed from given string s
    """

    return [ float( i ) for i in re.findall( r"[-+]?\d*\.\d+|[-+]?\d+", string2parse ) ]


def print_vars( vars2print: dict , save_dir = sys.stdout ):
    """
        Print out all the details of the variables to the standard output + file to save. 
    """

    # Iterate Through the dictionary for printing out the values. 
    for var_name, var_vals in vars2print.items( ):

        # Check if var_vals is a list or numpy's ndarray else just change it as string 
        if   isinstance( var_vals, ( list, np.ndarray ) ):
            
            # First, change the list to numpy array to make the problem easier 
            var_vals = np.array( var_vals ) if isinstance( var_vals, list ) else var_vals

            # If the numpy array is
            var_vals = np.array2string( var_vals.flatten( ), separator =', ', floatmode = 'fixed' )

        else:
            var_vals = str( var_vals )

        print( f'[{var_name}]: {var_vals}', file = save_dir )
