try:
    import cupy
    cupy.cuda.set_allocator(cupy.cuda.MemoryPool().malloc)
    xp = cupy
    print('Using CuPy.')
except ImportError:
    import numpy
    xp = numpy
    print('Using NumPy.')


class Parameter(object):

    def __init__(self, id, data):
        self.id = id
        self.data = data
