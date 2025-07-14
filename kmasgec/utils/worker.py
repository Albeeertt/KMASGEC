import multiprocessing as mp
from typing import Dict

class Worker(mp.Process):
    def __init__(self, stages, chunk, idx, dict_results: Dict[int, any], **kwargs):
        super(Worker, self).__init__(**kwargs)
        self._stages = stages
        self._chunk = chunk
        self._idx = idx
        self._dict_results = dict_results
        self.start()

    def run(self):
        chunk = self._chunk
        for processor in self._stages:
            chunk = processor(chunk)
        self._dict_results[self._idx] = chunk