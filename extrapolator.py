class extrapolator:
    """Extrapolate signal value, by keeping track of its derivatives over time"""
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
    deltas = []

    def __init__(self, src=None):
        # Optional-copy constructor
        if src is not None:
            self.deltas = src.deltas.copy()

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

    def get_value_time(self, delta_rank=0):
        """Retrieve specific derivative value and time"""
        return self.deltas[delta_rank]

    def get_value(self, delta_rank=0):
        """Retrieve specific derivative value"""
        return self.get_value_time(delta_rank)[0]

    def __next_times(self, time0):
        """Build a new 'times' vector by shifting the last one up"""
        return [time0] + [v[1] for v in self.deltas]

    def update(self, val0, time0):
        """Update 'deltas' based on a derivative value at specific moment"""
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

    def extrapolate(self, time0):
        """Extrapolate 'deltas' to specific moment (destructive)"""
        if len(self.deltas) < 1:
            return False

        # Advance delta's times, based on the required time0
        next_times = self.__next_times(time0)[:len(self.deltas)]

        # Extrapolate deltas at selected moments, assuming the last delta is constant
        self.deltas[-1] = (self.deltas[-1][0], next_times[-1])
        for i in reversed(range(0, len(self.deltas) - 1)):
            delta_t = time0 - self.deltas[i][1]
            delta_v = self.deltas[i + 1][0] * delta_t
            self.deltas[i] = (self.deltas[i][0] + delta_v, next_times[i])
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
                ret = self.extrapolate(time)
                if not ret:
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
