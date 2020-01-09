import multiprocessing, pathos

# A class that switches between serial and parallel mappers
# Also deletes parallel mapper when class is deleted
class GenericMapper():
    def __init__(self, serial, nCore=None):
        self.serial = serial
        self.pid = multiprocessing.current_process().pid

        if serial:
            self.nCore = 1
        else:
            self.nCore = nCore if nCore is not None else pathos.multiprocessing.cpu_count() - 1
            self.pool = pathos.multiprocessing.ProcessingPool(self.nCore)
            # self.pool = multiprocessing.Pool(self.nCore)

    # def __del__(self):
    #     if not self.serial:
    #         self.pool.close()
    #         self.pool.join()

    def map(self, f, x):
        print("----Root process", self.pid, "started task on", self.nCore, "cores----")
        rez = list(map(f, x)) if self.serial else self.pool.map(f, x)
        print("----Root process", self.pid, "finished task")
        return rez
