"""Approximator tests"""
import math
import poly_approx

FIRST_POLY = [0,0,1]        # x^2
SECOND_POLY = [0,0,-2]      # -2x^2

SWITCH_ITERATION = 8
TOTAL_ITERATIONS = 20
GAP_THRESHOLD = 3

def repr_polynomial(obj: poly_approx.approximator, off:float or None=None) -> str:
    t, var = (0, 'x') if off is None else (off, '(x-%s)'%off)
    coefs = obj.copy().get_poly_coefs(t)
    def flt_fmt(f): return ('%+10f'%f).rstrip('0')
    return ' '.join(['%s%s^%d'%(flt_fmt(c),var,i) if i else flt_fmt(c)
            for i, c in enumerate(coefs) if c][::-1])

def test_reduce() -> None:
    """Approximation simplification tests"""
    approx_obj = poly_approx.approximator()
    src_obj = poly_approx.approximator()
    src_obj.from_poly_coefs(FIRST_POLY)
    tmp_obj = poly_approx.approximator()

    time = 0
    for idx in range(TOTAL_ITERATIONS):
        print('%d:%3g -> %g'%(idx, time, src_obj.extrapolate(time)))

        approx_val = approx_obj.approximate(src_obj.get_value(), time)
        assert approx_val is not None

        # Drop too high-rank deltas
        if approx_val > GAP_THRESHOLD:
            print('* Drop %d deltas'%(approx_val - GAP_THRESHOLD))
            # Keep the threshold and the last dropped time
            approx_obj.reduce(GAP_THRESHOLD - approx_val, keep_time=True)

        # Do shock: switch to another function, by preserving some low-rank derivatives
        if idx == SWITCH_ITERATION:
            tmp_obj.from_poly_coefs(SECOND_POLY, time)
            print('* Switch from "%s" ("%s" plus "%s")'%(repr_polynomial(src_obj),
                    repr_polynomial(src_obj, time), repr_polynomial(tmp_obj, time)))
            tmp_obj.align_times(src_obj)
            src_obj += tmp_obj
            print('    to "%s" ("%s")'%(repr_polynomial(src_obj),
                    repr_polynomial(src_obj, time)))

        # Adjust 'src_obj' to the 'approx_obj' intervals
        tmp_obj = src_obj.copy().align_times(approx_obj)
        # Display the approximated deltas
        tmp_iter = iter(tmp_obj)
        for i, (val, t) in enumerate(approx_obj):
            src_vt = next(tmp_iter, None)
            if src_vt:
                assert src_vt[1] == t
                ref = '(ok)' if src_vt[0] == val else '(ref %8g)'%src_vt[0]
            else:
                ref = '?'
            print('    %d:%3g ->%8g %s'%(i, t, val, ref))
        print('  "%s" (approx val %d)'%(repr_polynomial(approx_obj), approx_val))

        # Check for "shock" function change
        first, num = approx_obj.find_gap(GAP_THRESHOLD)
        if num is not None:
            tmp_obj = approx_obj.split_at_gap(first + num)
            print('* Shock detected at', approx_obj.get_value_time(-1)[1])
            #FIXME: Some made-up rounding
            tmp_obj.reduce(min_val=1e-15, as_deriv=False)

            print('- Split into: "%s", after %f, before: "%s"'%(repr_polynomial(approx_obj),
                    approx_obj.get_value_time(-1)[1],
                    repr_polynomial(tmp_obj)))

        time += 1 + idx%4

#
# For non-pytest debugging
#
if __name__ == '__main__':
    test_reduce()
