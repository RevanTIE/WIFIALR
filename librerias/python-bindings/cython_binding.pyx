""" Ejemplo de Cython con C """

cdef extern from "cmult.h":
    float cmult(int int_param, float float_param)

def pymult(int_param, float_param):
    return cmult(int_param, float_param)
