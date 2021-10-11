class extrapolator:
    """Extrapolate signal value, by keeping track of its derivatives over time"""
    # Vector of the singal value deltas (val, time):
    # 0 - values itself (zero-derivative):
    #       delta0_next: value
    #       time0_next: time
    # 1 - first delta:
    #       delta1_next: (delta0_next - delta0_last) / (time0_next - time0_last)
    #       time1_next: (time0_next + time0_last) / 2
    # 2 - etc.
    deltas = []

    def __init__(self, src=None):
        # Optional-copy constructor
        if src is not None:
            self.deltas = src.deltas.copy()

    def __next_times(self, time, delta_rank=0, new_len=None):
        """Select new time vector, based on a specific derivative"""
        if new_len is None:
            new_len = len(self.deltas)
        next_times = [None] * new_len
        next_times[delta_rank] = time

        # Update toward differentials
        for i in range(delta_rank, new_len - 1):
            next_times[i + 1] = (next_times[i] + self.deltas[i][1]) / 2

        # Update toward intergrals, when delta_rank > 0
        for i in reversed(range(0, delta_rank)):
            next_times[i] = 2 * next_times[i + 1] - self.deltas[i][1]

        return next_times

    def update(self, val, time, delta_rank=0):
        """Update 'deltas' based on a derivative value at specific moment"""
        # Ensure there is a slot for the 'delta_rank' and an extra one after the extiting
        next_deltas = [None] * (1 + max(len(self.deltas), delta_rank))
        next_deltas[delta_rank] = val

        # Advance delta's times, based on the required time0
        next_times = self.__next_times(time, delta_rank, len(next_deltas))

        # Update toward differentials
        for i in range(delta_rank, len(self.deltas)):
            delta_v = next_deltas[i] - self.deltas[i][0]
            delta_t = next_times[i] - self.deltas[i][1]
            next_deltas[i + 1] = delta_v / delta_t
        # Drop the extra delta-rank, if zero
        if delta_rank < len(next_deltas) - 1 and next_deltas[-1] == 0.:
            next_deltas = next_deltas[:-1]

        # Update toward intergrals, when delta_rank > 0
        for i in reversed(range(0, delta_rank)):
            delta_t = next_times[i] - self.deltas[i][1]
            next_deltas[i] = self.deltas[i][0] + next_deltas[i + 1] * delta_t

        # Make the new vector actual one
        self.deltas = list(zip(next_deltas, next_times))
        return True

    def extrapolate(self, time0):
        """Generate extrapolated 'deltas' object at specific moment"""
        if len(self.deltas) < 1:
            return False

        # Advance delta's times, based on the required time0
        next_times = self.__next_times(time0)

        # Extrapolate deltas at selected moments, assuming the last delta is constant
        self.deltas[-1] = (self.deltas[-1][0], next_times[-1])
        for i in reversed(range(0, len(self.deltas) - 1)):
            delta_t = next_times[i] - self.deltas[i][1]
            delta_v = self.deltas[i + 1][0] * delta_t
            self.deltas[i] = (self.deltas[i][0] + delta_v, next_times[i])
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
