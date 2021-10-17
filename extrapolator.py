class extrapolator:
    """Extrapolate signal value, by keeping track of its derivatives over time"""
    # Vector of the signal value deltas (val, time):
    # 0 - values itself (zero-derivative):
    #       delta0_next: value
    #       time0_next: time
    # 1 - first mean derivative for the interval 'time0_last' to 'time0_next':
    #       delta1_next: 1 * (delta0_next - delta0_last) / (time0_next - time0_last)
    #       time1_next: time0_last
    # 2 - second mean derivative for the interval 'time1_last' to 'time0_next'):
    #       delta2_next: 2 * (delta1_next - delta1_last) / (time0_next - time1_last)
    #       time2_next: time1_last
    # 3 - etc.
    #
    # Note that, all mean derivative intervals end at 'time0_next' and each one is
    # included in the next
    deltas = []

    def __init__(self, src=None):
        # Optional-copy constructor
        if src is not None:
            self.deltas = src.deltas.copy()

    def __next_times(self, time, delta_rank=0):
        """Select new time vector, based on a specific derivative"""
        # Update toward intergrals, when delta_rank > 0
        next_times = [time] * (delta_rank + 1)
        assert delta_rank == 0, 'TODO: Use non-zero deltas toward integrals'

        # Update toward differentials
        next_times += [v[1] for v in self.deltas[delta_rank:]]
        return next_times

    def update(self, val, time, delta_rank=0):
        """Update 'deltas' based on a derivative value at specific moment"""
        # Advance delta's times, based on the required time0
        next_times = self.__next_times(time, delta_rank)

        # Initial next-delta array
        next_deltas = [None] * len(next_times)
        next_deltas[delta_rank] = val

        # Update toward differentials
        for i in range(delta_rank, len(self.deltas)):
            delta_v = next_deltas[i] - self.deltas[i][0]
            delta_t = time - self.deltas[i][1]
            if delta_t == 0:
                break   # Leave None in 'next_deltas'
            next_deltas[i + 1] = (i + 1 - delta_rank) * delta_v / delta_t
        # Drop the extra delta-rank, if zero
        if delta_rank < len(next_deltas) - 1 and not next_deltas[-1]:
            next_deltas = next_deltas[:-1]

        # Update toward intergrals, when delta_rank > 0
        for i in reversed(range(0, delta_rank)):
            delta_t = next_times[0] - self.deltas[i][1]
            delta_v = next_deltas[i + 1] * delta_t / (i + 1)
            next_deltas[i] = self.deltas[i][0] + delta_v * delta_t

        # Make the new vector actual one
        self.deltas = list(zip(next_deltas, next_times))
        return all(d is not None for d in next_deltas)

    def extrapolate(self, time0):
        """Generate extrapolated 'deltas' object at specific moment"""
        if len(self.deltas) < 1:
            return False

        # Advance delta's times, based on the required time0
        next_times = self.__next_times(time0)[:len(self.deltas)]

        # Extrapolate deltas at selected moments, assuming the last delta is constant
        self.deltas[-1] = (self.deltas[-1][0], next_times[-1])
        for i in reversed(range(0, len(self.deltas) - 1)):
            delta_t = time0 - self.deltas[i][1]
            delta_v = self.deltas[i + 1][0] * delta_t / (i + 1)
            self.deltas[i] = (self.deltas[i][0] + delta_v, next_times[i])
        return True

    def make_derivs(self, time=None, delta_rank=None):
        """Convert mean deltas to derivatives at specific moment"""
        if time is None:
            time = self.deltas[0][1]
        if delta_rank is None:
            delta_rank = len(self.deltas) - 1
        # Collapse the first 'delta_rank' mean delta intervals to zero
        for i in range(delta_rank + 1):
            if self.deltas[i][1] != time:
                ret = self.extrapolate(time)
                if not ret:
                    return False
        return True

    def __iter__(self):
        """Iterator over derivative values and times"""
        for i in range(self.num_deltas()):
            yield self.get_value_time(i)

    def num_deltas(self):
        """Get the number of derivatives"""
        return len(self.deltas)

    def get_value_time(self, delta_rank=0):
        """Retrieve specific derivative value and time"""
        return self.deltas[delta_rank]

    def get_value(self, delta_rank=0):
        """Retrieve specific derivative value"""
        return self.get_value_time(delta_rank)[0]
