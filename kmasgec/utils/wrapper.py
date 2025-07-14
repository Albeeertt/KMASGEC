from functools import wraps, partial

class Wrapper():
    def make_adapter(self, 
                     function,
                     input_selector, # =lambda chunk: (chunk,),
                     output_selector): # =lambda result, chunk: result if result is not None else chunk):
        
        adapter = partial(self._wrapped, function, input_selector, output_selector)
        wraps(function)(adapter)

        return adapter
    
    def _wrapped(self, function, input_selector, output_selector, chunk):
        args = input_selector(chunk)
        print("argumentos recibidos",args)
        print("Forma ideal: ", *args)
        res = function(*args)
        return output_selector(res, chunk)

    def tuple_chunk_todo(self, chunk):
        return chunk[0], chunk[1]
    
    def tuple_chunk_primero(self, chunk):
        return chunk[0],

    def tuple_chunk(self, chunk):
        return chunk,

    def no_more_tuples(self, chunk):
        return chunk


    def output_res_chunk1(self, res, chunk):
        return res, chunk[1]
    
    def output_res(self, res, chunk):
        return res
