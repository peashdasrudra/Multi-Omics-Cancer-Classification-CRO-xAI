import numpy as np
import random
import os

def set_seed(seed=42):
    """
    Locks the random seed across multiple libraries to ensure 100% reproducible results.
    This is critical for scientific validity in a thesis.
    """
    # Set numpy random seed
    np.random.seed(seed)
    
    # Set python built-in random seed
    random.seed(seed)
    
    # Set python hash seed for dictionary/set order reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to: {seed}")
