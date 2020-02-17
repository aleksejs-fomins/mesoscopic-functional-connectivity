import multiprocessing
import jpype as jp
import sys

from codes.lib.sys_lib import time_now_as_str, mem_now_as_str


# def parallelJVMShutdown(i):
#     if jp.isJVMStarted():
#         jp.shutdownJVM()
#         print("Had to shut down JVM")
#     return 0


def jpype_sync_thread(func):
    '''
    :param   func:  A thread function that may be using jpype
    :return: decorator for that function

    Purpose is to avoid memory leaks by synchronizing python threads with JAVA
    '''

    def attach_thread(*args, **kwargs):
        if jp.isJVMStarted():
            if jp.isThreadAttachedToJVM():
                print("This Thread is already attached")
            else:
                print("Attaching Thread")
                jp.attachThreadToJVM()

        rez = func(*args, **kwargs)

        if jp.isJVMStarted():
            if jp.isThreadAttachedToJVM():
                print("Detaching Thread")
                jp.detachThreadFromJVM()
            else:
                print("This thread was never attached")

            #jp.shutdownJVM()

        return rez

    return attach_thread


def redirect_stdout(func):
    '''
    :param   func:  A function that prints anything
    :return: decorator for that function

    Pipe all output of the function into a log file
    '''

    def inner(*args, **kwargs):
        procId = multiprocessing.current_process().pid

        orig_stdout = sys.stdout
        fname = 'log_' + str(procId) + '.txt'
        with open(fname, 'w') as f:
            print("Redirecting STDOUT to", fname, flush=True)
            sys.stdout = f
            rez = func(*args, **kwargs)
            sys.stdout = orig_stdout
            print("Resumed STDOUT", flush=True)

        return rez

    return inner


# Print time, memory usage, process id and first argument before and after the function
def time_mem_1starg(func):
    def inner(*args, **kwargs):
        procId = multiprocessing.current_process().pid
        print(time_now_as_str(), "proc:", procId, "mem:", mem_now_as_str(), "arg:", args[0], "started")

        rez = func(*args, **kwargs)

        print(time_now_as_str(), "proc:", procId, "mem:", mem_now_as_str(), "arg:", args[0], "finished")
        return rez

    return inner