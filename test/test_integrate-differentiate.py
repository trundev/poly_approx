"""Extrapolator integrate and differentiate test"""
import random
import math
import poly_approx

#
# Power of 4
#
if True:
    base_fns = (
        lambda x: x**4 - 2*x**3 + 3*x**2 - 4*x + 5,
        lambda x: 4*x**3 - 6*x**2 + 6*x - 4,
        lambda x: 12*x**2 - 12*x + 6,
        lambda x: 24*x - 12,
        lambda x: 24
    )
if True:
    base_fns = (
        lambda x: x**4,
        lambda x: 4*x**3,
        lambda x: 12*x**2,
        lambda x: 24*x,
        lambda x: 24
    )
#
# Power of 3
#
if False:
    base_fns = (
        lambda x: x**3 - 2*x**2 + 3*x - 4,
        lambda x: 3*x**2 - 4*x + 3,
        lambda x: 6*x - 4,
        lambda x: 6
    )
if False:
    base_fns = (
        lambda x: x**3,
        lambda x: 3*x**2,
        lambda x: 6*x,
        lambda x: 6
    )
#
# Power of 2
#
if False:
    base_fns = (
        lambda x: x**2,
        lambda x: 2*x,
        lambda x: 2
    )
#
# Power of 1
#
if False:
    def base_fn(x):
        return 3*x
    def base_fn_d(x):
        return 3
    base_fns = base_fn, base_fn_d
#
# Power of 0
#
if False:
    base_fns = (lambda x: 2,)

def polynomial_to_str(coefs: list, var: str='x') -> str:
    return ' + '.join(['%s*%s**%d'%(c,var,i) if i else str(c)
            for i, c in enumerate(coefs) if c][::-1])

def repr_polynomial(obj, off=None):
    t,var = (0,'x') if off is None else (off,'(x-%s)'%off)
    return polynomial_to_str(obj.copy().get_poly_coefs(t), var)

EPSILON = 1e-9

def test_main() -> None:
    """Approximation and extrapolation tests on irregular intervals"""
    obj = poly_approx.approximator()

    seed = random.random()
    print(f'{seed=}')
    random.seed(seed)

    total_cnt, total_error = 0, 0.
    e_obj = obj
    d_obj = poly_approx.approximator()
    time = 0
    result = []
    for i in range(10):
        # Key point (start from the original object) or continue extrapolation
        if e_obj.num_deltas() < len(base_fns):      # Qubic functions: upto fifth point
            pfix = '-Key-'
            e_obj = obj.copy()
        else:
            pfix = '-'

        # Keep the step divisor a power of 2 to avoid float rounding errors
        step = random.randint(-4000, 4000) / 2**10
        time += step
        print(pfix, '%f (%+f)'%(time, step))

        # Approximation on 'obj' and extrapolation on 'e_obj'
        assert e_obj is not obj, 'Must extrapolate on separate object'
        res = obj.approximate(base_fns[0](time), time)
        assert res is not None, 'Approximation failed'
        if res > 0:
            assert res + obj.num_deltas(0) == obj.num_deltas(), 'Incorrect approximation result'
            obj.reduce(-res, keep_time=True)
        e_res = e_obj.extrapolate(time, keep_time=True)

        #
        # Differentiate object
        #
        if obj.num_deltas() > 1:
            t = time - 1.5
            diff_obj = obj.copy()
            diff_obj.differentiate(t)
            if res > 0:
                assert all(fn(t) == diff_obj.get_value(i, True) for i, fn in enumerate(base_fns[1:]))
            diff_obj.extrapolate(time)
            if res > 0:
                assert base_fns[1](time) == diff_obj.get_value(0, True)
            print(f'Differential {diff_obj.get_value_time()[1]} = {diff_obj.get_value()} polynomial: d({repr_polynomial(obj)})/dx = {repr_polynomial(diff_obj)}')

        #
        # Integrate differential object (how to update higher 'delta_rank')
        #
        if d_obj.approximate(base_fns[1](time), time) > 0:
            t = time - 1.5
            int_obj = d_obj.copy()
            int_obj.reduce()
            int_obj.integrate(base_fns[0](t), t)
            assert all(fn(t) == int_obj.get_value(i, True) for i, fn in enumerate(base_fns))
            int_obj.extrapolate(time)   # or int_obj.make_derivs(time,0)
            assert abs(base_fns[0](time) - int_obj.get_value(0, True)) <= EPSILON
            coefs = int_obj.copy()
            coefs.reduce(obj.num_deltas())
            coefs = coefs.get_poly_coefs()
            print(f'Integral {t} -> {time} = {int_obj.get_value()} polynomial: int{{ {repr_polynomial(d_obj)} }}dx = {polynomial_to_str(coefs)}')
            for act, exp in zip(obj.copy().get_poly_coefs(), coefs):
                assert abs(act - exp) <= EPSILON

        # Input function and its averaged derivatives
        res = ''
        for i, (v, t) in enumerate(obj):
            if i == 0:
                res += '%9.2f:%10.2f'%(t, base_fns[i](t))
                t0 = t
            else:
                if i <= len(base_fns):
                    int_div = (t0 - t) * math.factorial(i)
                    avg_deriv = (base_fns[i - 1](t0) - base_fns[i - 1](t)) / int_div
                else:
                    avg_deriv = float('nan')
                res += ' |%6.2f%-+.2f:%8.2f'%(t, t0 - t, avg_deriv)
        print(' ', res)
        # Current 'extrapolator' object and the extrapolation started at '-Key-'
        print('<', ' |'.join('%9.2f:%10.2f'%d[::-1] for d in obj))
        print('>', ' |'.join('%9.2f:%10.2f'%d[::-1] for d in e_obj))
        if e_res is not None:
            result.append(f'{time:6.2f}: {obj.get_value():8.2f} {e_obj.get_value():8.2f}, '
                    f'diff {e_obj.get_value() - obj.get_value():6.2f}%s'%(
                        f' {100 * (e_obj.get_value() - obj.get_value()) / obj.get_value():6.2f}%'
                        if obj.get_value() else ''))
            # Extrapolation is incorrect during initial intervals
            if e_obj.num_deltas() >= len(base_fns):     # Qubic functions: start at fifth point
                total_cnt += 1
                total_error += e_obj.get_value() - obj.get_value()
        else:
            result.append(f'{time:6.2f}: {obj.get_value():8.2f}')

    coefs = obj.copy().get_poly_coefs()
    print(4*'-', 'polynomial:', polynomial_to_str(coefs))
    print('\n'.join(result))
    assert total_error == 0 and total_cnt, 'Accumulater error %g from %d attempts'%(total_error, total_cnt)

#
# For non-pytest debugging
#
if __name__ == '__main__':
    test_main()
