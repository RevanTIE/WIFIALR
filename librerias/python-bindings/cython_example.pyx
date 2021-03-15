cimport milibreria

""" Example cython interface definition """

def pymult(int_param, float_param):
    return milibreria.cppmult(int_param, float_param)
