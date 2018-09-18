try:
    import cupy
    import cupyx
    cupy.cuda.set_allocator(cupy.cuda.MemoryPool().malloc)
    xp = cupy
    xpx = cupyx
    print('Using CuPy.')
except ImportError:
    import numpy
    xp = numpy
    xpx = None
    print('Using NumPy.')


class Parameter(object):

    def __init__(self, id, data):
        self.id = id
        self.data = data
