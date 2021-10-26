"""Simple polynomial approximation tool"""
import math

class approximator:
    """Approximate signal value, by keeping track of its derivatives over time"""
    # Vector of the signal value deltas (val, time):
    # 0 - values itself (zero-derivative):
    #       delta0: value
    #       time0: time
    # 1 - first mean derivative for the interval 'time1' to 'time0':
    #       delta1: 1 * (delta0 - delta0_last) / (time0 - time0_last)
    #       time1: time0_last
    # 2 - second mean derivative for the interval 'time2' to 'time0'):
    #       delta2: 2 * (delta1 - delta1_last) / (time0 - time1_last)
    #       time2: time1_last
    # 3 - etc.
    #
    # Note that, all mean derivative intervals end at 'time0' and each one is
    # included in the next
    deltas = None

    def __init__(self, src=None):
        # Optional-copy constructor
        self.deltas = [] if src is None else src.deltas.copy() 

    def copy(self):
        """Duplicate the object"""
        return __class__(self)

    def __iter__(self):
        """Iterator over derivative values and times"""
        for d_x in self.deltas:
            yield d_x

    def num_deltas(self):
        """Get the number of derivatives"""
        return len(self.deltas)

    def get_value_time(self, delta_rank=0, as_deriv=False):
        """Retrieve specific mean delta/derivative value and time"""
        if as_deriv:
            return self.deltas[delta_rank][0] * math.factorial(delta_rank), self.deltas[delta_rank][1]
        return self.deltas[delta_rank]

    def get_value(self, delta_rank=0, as_deriv=False):
        """Retrieve specific mean delta/derivative value"""
        return self.get_value_time(delta_rank, as_deriv)[0]

    def __next_times(self, time0):
        """Build a new 'times' vector by shifting the last one up"""
        return [time0] + [v[1] for v in self.deltas]

    def approximate(self, val0, time0):
        """Update 'deltas' based on a signal value at specific moment"""
        # Advance delta's times, based on the required time0
        next_times = self.__next_times(time0)

        # Initial next-delta array
        next_deltas = [None] * len(next_times)
        next_deltas[0] = val0

        # Update toward differentials
        for i in range(len(self.deltas)):
            delta_t = time0 - self.deltas[i][1]
            if delta_t == 0:
                break   # Leave None in 'next_deltas'
            delta_v = next_deltas[i] - self.deltas[i][0]
            next_deltas[i + 1] = delta_v / delta_t
        # Drop the extra delta-rank, if zero
        if 1 < len(next_deltas) and not next_deltas[-1]:
            next_deltas = next_deltas[:-1]
            ret = False
        else:
            ret = True      # This value is independent from others

        # Make the new vector actual one
        self.deltas = list(zip(next_deltas, next_times))
        return None if any(d is None for d in next_deltas) else ret

    def __add__(self, obj):
        """Sum of approximations with compatible intervals"""
        res_obj = __class__()
        for (v, t), (v1, t1) in zip(self.deltas, obj.deltas):
            #assert t == t1, 'The intervals do not match (%d, %d)'%(t, t1)
            if t != t1:
                return None
            res_obj.deltas.append((v + v1, t))

        # Append the remainder
        if len(self.deltas) != len(obj.deltas):
            rem_deltas = self.deltas if len(self.deltas) > len(obj.deltas) else obj.deltas
            res_obj.deltas += rem_deltas[len(res_obj.deltas):]
        return res_obj

    def __neg__(self):
        """Negate the approximation"""
        res_obj = __class__()
        res_obj.deltas = [(-v, t) for v, t in self.deltas]
        return res_obj

    def __sub__(self, obj):
        """Difference between approximations with compatible intervals"""
        return self.__add__(obj.__neg__())

    #
    # The functions below, marked as (destructive),
    # must not be used on an object, where the "approximation" is in progress.
    # These transformations must be applied on a copy of such object.
    #
    def extrapolate(self, time0):
        """Extrapolate 'deltas' to specific moment (destructive)"""
        if len(self.deltas) < 1:
            return None

        # Advance delta's times, based on the required time0
        next_times = self.__next_times(time0)[:len(self.deltas)]

        # Extrapolate deltas at selected moments, assuming the last delta is constant
        self.deltas[-1] = (self.deltas[-1][0], next_times[-1])
        for i in reversed(range(0, len(self.deltas) - 1)):
            delta_t = time0 - self.deltas[i][1]
            delta_v = self.deltas[i + 1][0] * delta_t
            self.deltas[i] = (self.deltas[i][0] + delta_v, next_times[i])
        return self.deltas[0][0]

    def differentiate(self, time0=None):
        """Shift the 'deltas' down to match the differential function (destructive)"""
        if not self.make_derivs(time0):
            return False
        self.deltas = [(d * (i + 1), t) for i, (d, t) in enumerate(self.deltas[1:])]
        return True

    def integrate(self, val0, time0=None):
        """Shift the 'deltas' up to match the integral function (destructive)"""
        if not self.make_derivs(time0):
            return False
        int_deltas = [(d / (i + 1), t) for i, (d, t) in enumerate(self.deltas)]
        self.deltas = [(val0, self.deltas[0][1])] + int_deltas
        return True

    def make_derivs(self, time=None, delta_rank=None):
        """Convert mean deltas to derivatives at specific moment (destructive)"""
        if time is None:
            time = self.deltas[0][1]
        if delta_rank is None:
            delta_rank = len(self.deltas) - 1
        if delta_rank < 0:
            return False
        # Collapse the first 'delta_rank' mean delta intervals to zero
        for i in range(delta_rank + 1):
            if self.deltas[i][1] != time:
                if self.extrapolate(time) is None:
                    return False
        return True

    def get_poly_coefs(self, time=0):
        """Calculate polynomial coefficients (destructive)"""
        if not self.make_derivs(time):
            return None
        return [d[0] for i, d in enumerate(self.deltas)]

    def from_poly_coefs(self, coefs, time=0):
        """Initialize 'deltas' from polynomial coefficients (destructive)"""
        self.deltas = [(c, time) for i, c in enumerate(coefs)]

    def reduce(self, max_rank=None, min_val=0, as_deriv=False):
        """Cut highest-rank derivatives: by number and/or by minimal value (destructive)"""
        for i in reversed(range(1, len(self.deltas[:max_rank]))):
            val = self.deltas[i][0]
            if as_deriv:
                val *= math.factorial(i)
            if math.fabs(val) > min_val:
                max_rank = i + 1
                break
        self.deltas = self.deltas[:max_rank]
