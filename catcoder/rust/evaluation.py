import os
import json

from datasets import load_from_disk

from metrics import CratePassK
from inference import Model, OpenAIModel, VllmClientModel
from util import (
    build_prompt,
    remove_markdown, 
    fix_fragmented_code, 
    truncate_generation,
)

class Benchmark:
    '''
    Base class for benchmarks. The benchmark is responsible for:
    - Loading data from HF dataset
    - Generating code from model and postprocessing
    - Evaluating the generated code
    '''
    def __init__(self, model: Model, name: str, n=10, k=[1,3,5], cache=False):
        '''
        - `n` and `k`, refer to https://arxiv.org/abs/2107.03374 for details.
        - `cache`, whether to load cached results from disk. 
        '''
        self.name = name
        self.model = model
        self.n = n
        self.k = k
        self.cache = cache
        self.data = load_from_disk(f'./dataset/{self.name}')
        self.postprocs = [truncate_generation, remove_markdown, fix_fragmented_code]
        os.makedirs(f'results/{self.name}', exist_ok=True)
    
    @property
    def cache_file(self):
        return f'results/{self.name}/{self.model.info}_n{self.n}.json'
    
    def _from_hf_data(self, data):
        raise NotImplementedError()

    def _postprocess_code(self, code):
        for fn in self.postprocs:
            code = fn(code)
        return code

    def _model(self, prompt):
        return self._postprocess_code(self.model.infer(prompt))
    
    def _codegen(self, data, *_, **__):
        '''
        Use model to generate code. To support alternative models or pipelines, 
        override this method in subclasses.
        '''
        codes = []
        
        for _ in range(self.n):
            codegen_prompt = build_prompt(self._from_hf_data(data), True)
            code = self._model(codegen_prompt)
            if not code.startswith('fn') and not code.startswith('pub fn'):
                code = data['signature'] + ' ' + code
            codes.append(code)
        
        return codes

    def _batched_codegen(self):
        fn_codes = []

        if self.cache and os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                d = json.load(f)
                fn_codes = d['fn_codes']
                assert d['n'] == self.n
                assert d['benchmark'] == self.name
        else:
            for idx, data in enumerate(self.data):
                print(f'Processing {idx}...')
                results = self._codegen(data)
                assert len(results) == self.n
                fn_codes.append(results)
        
        return fn_codes
    
    def _evaluate(self):
        '''
        Evaluate the generated code using unbiased pass@k metric.
        '''
        raise NotImplementedError()
    
    def _dump_cache(self, metric, fn_codes):
        if self.cache and os.path.exists(self.cache_file):
            print('Skipping cache dump, so as not to overwrite existing cache.')
            return
        with open(self.cache_file, 'w+') as f:
            d =  {
                'benchmark': self.name,
                'lastest_eval': metric.to_dict(),
                'n': self.n,
                'k': self.k,
                'fn_codes': fn_codes,
            }
            json.dump(d, f, indent=2)
        print(f'Results dumped to {self.cache_file}.')

    def evaluate(self):
        self._dump_cache(*self._evaluate())


class RustEval(Benchmark):
    def __init__(self, model: Model, n=10, cache=False):
        super().__init__(model, 'rusteval', n=n, cache=cache)
        self.crates_base = './crates'
    
    def _evaluate(self):
        print(f'Running {self.name} benchmark with n={self.n} ...')
        fn_codes = self._batched_codegen()
        metric = CratePassK(self.n, self.k, fn_codes, self.crates_base, self.data)
        print(metric)
        return metric, fn_codes

class RustEvalCatCoder(RustEval):
    def __init__(self, model: Model, n=10, cache=False):
        super().__init__(model, n, cache)
        self.name = 'rusteval_xc'
        os.makedirs(f'results/{self.name}', exist_ok=True)

    def _from_hf_data(self, data):
        return {
            'focal_fn_signature': data['signature'],
            'docstring': data['docstring'],
            'focal_ctx': data['extended_context'],
            'rag_data': data['rag_data'],
        }

class RustEvalInFile(RustEval):
    def __init__(self, model: Model, n=10, cache=False):
        super().__init__(model, n, cache)
        self.name = 'rusteval_if'
        os.makedirs(f'results/{self.name}', exist_ok=True)

    def _from_hf_data(self, data):
        return {
            'focal_fn_signature': data['signature'],
            'docstring': data['docstring'],
            'focal_ctx': data['infile_context'],
            'rag_data': '',
        }

class RustEvalRepoCoder(RustEval):
    def __init__(self, model: Model, n=10, cache=False):
        super().__init__(model, n, cache)
        self.name = 'rusteval_repo'
        os.makedirs(f'results/{self.name}', exist_ok=True)

    def _from_hf_data(self, data):
        return {
            'focal_fn_signature': data['signature'],
            'docstring': data['docstring'],
            'focal_ctx': '',
            'rag_data': data['repocoder_data'],
        }
    
class RustEvalVanilla(RustEval):
    def __init__(self, model: Model, n=10, cache=False):
        super().__init__(model, n, cache)
        self.name = 'rusteval_basic'
        os.makedirs(f'results/{self.name}', exist_ok=True)

    def _from_hf_data(self, data):
        return {
            'focal_fn_signature': data['signature'],
            'docstring': data['docstring'],
            'focal_ctx': '',
            'rag_data': '',
        }
    
class RustEvalWithoutContext(RustEval):
    def __init__(self, model: Model, n=10, cache=False):
        super().__init__(model, n, cache)
        self.name = 'rusteval-tc'
        os.makedirs(f'results/{self.name}', exist_ok=True)

    def _from_hf_data(self, data):
        return {
            'focal_fn_signature': data['signature'],
            'docstring': data['docstring'],
            'focal_ctx': '',
            'rag_data': data['rag_data'],
        }

class RustEvalWithoutRetrieval(RustEval):
    def __init__(self, model: Model, n=10, cache=False):
        super().__init__(model, n, cache)
        self.name = 'rusteval-cr'
        os.makedirs(f'results/{self.name}', exist_ok=True)

    def _from_hf_data(self, data):
        return {
            'focal_fn_signature': data['signature'],
            'docstring': data['docstring'],
            'focal_ctx': data['extended_context'],
            'rag_data': '',
        }

if __name__ == '__main__':
    model = VllmClientModel('codellama-13b')
    benchmark = RustEvalCatCoder(model, n=10, cache=False)
    benchmark.evaluate()
