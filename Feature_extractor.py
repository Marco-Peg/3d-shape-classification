import argparse

import igl
import scipy as sp
import numpy as np
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from scipy.spatial import distance

def features_extractor(classes,DNA_size=50):
    """Extract the features from the samples in the dataset. The type of features is the ShapeDNA.

    :param classes: list
        list of the classes
    :param DNA_size: int
        size of the feature vector
    """
    print("\nFeature extraction")
    
    for c in classes:
        print(c)
        for file in os.listdir(os.path.join('.', 'Dataset', c)):
            if file.endswith(".off"):
                print("\t", file, end=" ... ")
                # Load mesh
                v, f = igl.read_triangle_mesh(os.path.join('.', 'Dataset',c, file))
                M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
                v = v / np.sqrt(M.sum())

                # Compute Laplacian
                L = -igl.cotmatrix(v, f)
                M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
                # Compute EigenDecomposition
                try:
                    evals, evecs = sp.sparse.linalg.eigsh(L, DNA_size+2, M, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
                except:
                    evals, evecs = sp.sparse.linalg.eigsh(L + 1e-8 * sp.sparse.identity(v.shape[0]), DNA_size+2,
                                                          M, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
                # Shape DNA
                descriptors = evals[2:] / evals[1]
                # Save descriptor
                np.save(os.path.join('.', 'Dataset', c, file[:-4] + "_DNA"), descriptors, allow_pickle=False)

                print("done.")
    print("Finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Features extractor')
    parser.add_argument('-c', '--classes', nargs='+', help='<Required> Classes', required=True)
    parser.add_argument("-s", "--descriptor_size", default=100, type=int,
                        help="Size of the descriptor", dest="DNA_size")

    args = vars(parser.parse_args())
    features_extractor(**args)

