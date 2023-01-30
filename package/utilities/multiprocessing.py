import multiprocessing
import tqdm



class parallelizer(object):
    ### base class for multiprocessing
    def __init__(self, args, func, n_procs, desc):
        ### function initialization
        self.num_procs = n_procs
        self.args = args
        self.func = func
        self.desc = desc

    def start(self):
        pass

    def end(self):
        pass

    def run(args, func, num_procs, desc):
        return MultiThreading(args, func, num_procs, desc)
        ### run takes 4 arguments:
            # list of tup(args) for each spawned process
            # name of the function to be multiprocessed
            # number of process to spwan
            # description for the progression bar



def MultiThreading(args, func, num_procs, desc):
    results = []
    tasks = []
    for index,item in enumerate(args):
        task = (index, (func, item))
        ### every queue objects become indexable
        tasks.append(task)
    ### step needed to rethrieve correct results order
    results = start_processes(tasks, num_procs, desc)
    return results



def start_processes(inputs, num_procs, desc):
    ### this function effectively start multiprocess
    task_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()
    ### queue objects to manage to do args and results

    for item in inputs:
        ### inputs = [(index, (function_name, arg)), ...]
        task_queue.put(item)
        ### every item is passed to the task_queue

    pbar = tqdm.tqdm(total=len(inputs), desc=desc)
    ### progress bar is initialized
    
    for i in range(num_procs):
        multiprocessing.Process(target=worker, args=(task_queue, done_queue)).start()
        ### spawn (n_proc) worker function, that takes queue objects as args

    results = []
    for i in range(len(inputs)):
        results.append(done_queue.get())
        pbar.update(1)
        ### done_queue and progress bar update for each done object

    for i in range(num_procs):
        task_queue.put("STOP")
        ### to exit from each spawned process when task queue is empty

    results.sort(key=lambda tup: tup[0])    ### sorting of args[a] - args[z] results
    return [item[1] for item in map(list, results)] ### return a list with sorted results



def worker(input, output):
    ### input, output = task_queue, done_queue lists
    for seq, job in iter(input.get, "STOP"):
        ### seq = object index
        ### job = function name, args for function

        func, args = job
        result = func(*args)
        ### function is executed and return a result value

        ret_val = (seq, result)
        ### assign index to result value

        output.put(ret_val)
        ### (index, result) object is put in done_queue