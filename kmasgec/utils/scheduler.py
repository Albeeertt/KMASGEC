from multiprocessing import Manager
from worker import Worker
import logging



class Scheduler:
    def __init__(self, stages, chunker):
        self._stages = stages
        self._chunker = chunker
        self._logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def run(self, *args, **kwargs):
        chunks = self._chunker(*args, **kwargs)
        mgr = Manager()
        return_dict = mgr.dict()
        procs = []
        for idx, chunk in enumerate(chunks):
            worker = Worker(self._stages, chunk, idx, return_dict)
            procs.append(worker)

        for p in procs: p.join()

        chunks = [return_dict[key] for key in return_dict.keys()]
        return chunks
