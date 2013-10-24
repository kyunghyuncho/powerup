"""Objects for datasets serialized in the NumPy native format (.npy/.npz)."""
import functools
import numpy
from theano import config
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

class NpyLabeled(DenseDesignMatrix):
    """A dense dataset based on a single array stored as a .npy file."""
    """The last column of the 2-d array stores labels """
    def __init__(self, file, mmap_mode=None):
        """
        Creates an NpzDataset object.

        Parameters
        ----------
        file : file-like object or str
            A file-like object or string indicating a filename. Passed
            directly to `numpy.load`.
        mmap_mode : str, optional
            Memory mapping options for memory-mapping an array on disk,
            rather than loading it into memory. See the `numpy.load`
            docstring for details.
        """
        self._path = file
        self._loaded = False

    def _deferred_load(self):
        self._loaded = True
        loaded = numpy.load(self._path)
        assert isinstance(loaded, numpy.ndarray), (
            "single arrays (.npy) only"
        )
        assert len(loaded.shape) == 2
        super(NpyLabeled, self).__init__(X=loaded[:,:-1], y=loaded[:,-1])

    @functools.wraps(DenseDesignMatrix.get_design_matrix)
    def get_design_matrix(self, topo=None):
        if not self._loaded:
            self._deferred_load()
        return super(NpyLabeled, self).get_design_matrix(topo)

    @functools.wraps(DenseDesignMatrix.get_topological_view)
    def get_topological_view(self, mat=None):
        if not self._loaded:
            self._deferred_load()
        return super(NpyLabeled, self).get_topological_view(mat)

    @functools.wraps(DenseDesignMatrix.iterator)
    def iterator(self, *args, **kwargs):
        # TODO: Factor this out of iterator() and into something that
        # can be called by multiple methods. Maybe self.prepare().
        if not self._loaded:
            self._deferred_load()
        return super(NpyLabeled, self).iterator(*args, **kwargs)

