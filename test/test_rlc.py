"""Test by using RLC circuit model with complex numbers"""
import math
import poly_approx

FIRST_POLY = [0,0,1]        # x^2

def repr_polynomial(obj, off=None):
    t, var = (0, 'x') if off is None else (off, '(x-%s)'%off)
    coefs = obj.copy().get_poly_coefs(t)
    def flt_fmt(f): return ('%+10f'%f).rstrip('0')
    return ' '.join(['%s%s^%d'%(flt_fmt(c),var,i) if i else flt_fmt(c) for i, c in enumerate(coefs) if c][::-1])

def euler_formula(euler_omega, t):
    return math,

def calculate_current(a, omega):
    return

def test_rlc():
    """Approximation simplification tests"""
    approx_obj = poly_approx.approximator()

if __name__ == '__main__':
    test_rlc()
