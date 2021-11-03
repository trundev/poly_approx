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
        for v_t in self.deltas:
            yield v_t

    def __reversed__(self):
        """Reverse iterator over derivative values and times"""
        for v_t in self.deltas[::-1]:
            yield v_t

    def num_deltas(self, min_val=None):
        """Get the number of derivatives (or actual ones)"""
        if min_val is None:
            return len(self.deltas)
        # Obtain the first "actual" value in reverse
        for i, (v, _) in enumerate(self.deltas[::-1]):
            if math.fabs(v) > min_val:
                return len(self.deltas) - i
        return 0

    def get_value_time(self, delta_rank=0, as_deriv=False):
        """Retrieve specific mean delta/derivative value and time"""
        if as_deriv:
            # Allow python-style negative indices
            if delta_rank < 0:
                delta_rank += len(self.deltas)
            return self.deltas[delta_rank][0] * math.factorial(delta_rank), self.deltas[delta_rank][1]
        return self.deltas[delta_rank]

    def get_value(self, delta_rank=0, as_deriv=False):
        """Retrieve specific mean delta/derivative value"""
        return self.get_value_time(delta_rank, as_deriv)[0]

    def __next_times(self, time0):
        """Build a new 'times' vector by shifting the last one up"""
        return [time0] + [t for _, t in self.deltas]

    def find_gap(self, min_ranks, max_val=0, as_deriv=False):
        """Locate the first continuous gap of negligible deltas"""
        first = ranks = 0
        for i, (v, _) in enumerate(self.deltas):
            if as_deriv:
                v *= math.factorial(i)
            if math.fabs(v) <= max_val:
                ranks += 1
            elif ranks < min_ranks:
                ranks = 0
                first = i + 1
            else:
                return first, ranks
        return first, None

    def approximate(self, val0, time0):
        """Update 'deltas' based on a signal value at specific moment"""
        # Advance delta's times, based on the required time0
        next_times = self.__next_times(time0)

        # Initial next-delta array
        next_deltas = [None] * len(next_times)
        next_deltas[0] = val0

        # Update toward differentials
        num_independ = 1
        for i, (v, t) in enumerate(self.deltas):
            delta_t = time0 - t
            if delta_t == 0:
                return None     # Leave the object intact
            delta_v = next_deltas[i] - v
            next_deltas[i + 1] = delta_v / delta_t
            if delta_v != 0:
                num_independ = i + 2

        # Make the new vector actual one
        self.deltas = list(zip(next_deltas, next_times))
        return len(self.deltas) - num_independ

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
    def extrapolate(self, time0, keep_time=False):
        """Extrapolate 'deltas' to specific moment (destructive)"""
        if len(self.deltas) < 1:
            return None

        # Advance delta's times, based on the required time0
        next_times = self.__next_times(time0)[:len(self.deltas)]

        # The last delta is assumed constant, thus its time-interval does not affect
        # this result. However, further operations might need the whole scope.
        if not keep_time:
            self.deltas[-1] = (self.deltas[-1][0], next_times[-1])
        # Update toward intergrals, starting from the next to last
        for i in reversed(range(0, len(self.deltas) - 1)):
            delta_t = time0 - self.deltas[i][1]
            delta_v = self.deltas[i + 1][0] * delta_t
            self.deltas[i] = (self.deltas[i][0] + delta_v, next_times[i])
        return self.deltas[0][0]

    def rewind(self):
        """Revert the result from the last approximate() invocation"""
        # The end of all intervals is in rank 0
        _, time0 = self.deltas[0]

        # Update toward differentials, like approximate()
        for i, (v, _) in enumerate(self.deltas[:-1]):
            # The time was moved into the upper rank, see __next_times()
            t = self.deltas[i + 1][1]
            delta_v = self.deltas[i + 1][0] * (time0 - t)
            self.deltas[i] = (v - delta_v, t)

        # Cut the last delta
        self.deltas = self.deltas[:-1]

    def differentiate(self, time0=None):
        """Shift the 'deltas' down to match the differential function (destructive)"""
        if not self.make_derivs(time0):
            return False
        self.deltas = [(v * (i + 1), t) for i, (v, t) in enumerate(self.deltas[1:])]
        return True

    def integrate(self, val0, time0=None):
        """Shift the 'deltas' up to match the integral function (destructive)"""
        if not self.make_derivs(time0):
            return False
        int_deltas = [(v / (i + 1), t) for i, (v, t) in enumerate(self.deltas)]
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
                if self.extrapolate(time, keep_time=False) is None:
                    return False
        return True

    def get_poly_coefs(self, time=0):
        """Calculate polynomial coefficients (destructive)"""
        if not self.make_derivs(time):
            return None
        return [v for v, _ in self.deltas]

    def from_poly_coefs(self, coefs, time=0):
        """Initialize 'deltas' from polynomial coefficients (destructive)"""
        self.deltas = [(c, time) for c in coefs]

    def reduce(self, max_rank=None, min_val=None, as_deriv=False, keep_time=False):
        """Cut highest-rank derivatives: by number and/or by minimal value (destructive)"""
        if keep_time:
            _, last_time = self.deltas[-1]
        if min_val is not None:
            # Obtain the first "actual" value in reverse, starting at "max_rank"
            for i in reversed(range(1, len(self.deltas[:max_rank]))):
                val = self.deltas[i][0]
                if as_deriv:
                    val *= math.factorial(i)
                if math.fabs(val) > min_val:
                    max_rank = i + 1
                    break
            else:
                max_rank = 0
        self.deltas = self.deltas[:max_rank]
        if keep_time:
            self.deltas[-1] = self.deltas[-1][0], last_time
        return len(self.deltas)

    def align_times(self, ref_obj):
        """Align time-intervals to the ones in 'ref_obj', allows __add__() (destructive)"""
        # Use the minimal set of times
        for _, t in reversed(ref_obj.deltas[:len(self.deltas)]):
            if self.extrapolate(t, keep_time=False) is None:
                return None
        return self

    def split_at_gap(self, split_rank):
        """Split object into two, by using reduce() and rewind() (destructive)"""
        rest_obj = self.copy()
        self.reduce(split_rank)

        # Temporarily drop the cut-out and rewind
        rest_obj -= self
        for _ in range(split_rank - 1):
            rest_obj.rewind()

        # Combine actual object by adding back the cut-out (must be aligned)
        align_obj = self.copy()
        if align_obj.align_times(rest_obj) is None:
            return None
        rest_obj += align_obj
        return rest_obj
