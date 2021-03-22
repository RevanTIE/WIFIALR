cimport milibreria

""" Ejemplo de Cython con C++ """

def pymult(int_param, float_param):
    return milibreria.cppmult(int_param, float_param)
