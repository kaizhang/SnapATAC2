import unittest
import imghdr
# import snapatac2 as snap
import numpy as np
import scanpy as sc
import os
from _qc import *
from _util import *

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

class TestPlot(unittest.TestCase):
    def test_tsse(self):
        """
        Test the tsse plot & saving
        """
        file_path = '/projects/ps-renlab/y8yuan/projects/my_snapatac2/count_matrix.h5ad'
        data = sc.read_h5ad(file_path)
        outpath = '/projects/ps-renlab/y8yuan/projects/my_snapatac2'
        tsse(data,True,outpath)
        save_path = outpath +'/tsse.png'    
        result = imghdr.what(save_path)
        expected = 'png'
        self.assertEqual(result, expected)


        
if __name__ == '__main__':
    unittest.main()






