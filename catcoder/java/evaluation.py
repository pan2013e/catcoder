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
    def __init__(self, model: Model, name: str, n=10, k=[1,3,5], cache=False):
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
        codes = []
        
        for _ in range(self.n):
            codegen_prompt = build_prompt(self._from_hf_data(data), True)
            code = self._model(codegen_prompt).strip()
            if not code.startswith('public') and \
                not code.startswith('private') and \
                    not code.startswith('protected') and \
                        not code.startswith('static') and \
                            not code.startswith('@'):
                code = data['focal_fn_signature'] + ' ' + code
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

class JavaEval(Benchmark):
    def __init__(self, model: Model, n=10, cache=False):
        super().__init__(model, 'javaeval', n=n, cache=cache)
    
    def _evaluate(self):
        print(f'Running {self.name} benchmark with n={self.n} ...')
        fn_codes = self._batched_codegen()
        metric = CratePassK(self.n, self.k, fn_codes, self.data)
        print(metric)
        return metric, fn_codes

class JavaEvalCatCoder(JavaEval):
    def __init__(self, model: Model, n=10, cache=False):
        super().__init__(model, n, cache)
        self.name = 'javaeval_xc'
        os.makedirs(f'results/{self.name}', exist_ok=True)

    def _from_hf_data(self, data):
        return {
            'focal_fn_signature': data['focal_fn_signature'],
            'docstring': data['docstring'],
            'focal_ctx': data['extended_context'],
            'rag_data': data['rag_data'],
        }

class JavaEvalInFile(JavaEval):
    def __init__(self, model: Model, n=10, cache=False):
        super().__init__(model, n, cache)
        self.name = 'javaeval_if'
        os.makedirs(f'results/{self.name}', exist_ok=True)

    def _from_hf_data(self, data):
        return {
            'focal_fn_signature': data['focal_fn_signature'],
            'docstring': data['docstring'],
            'focal_ctx': data['infile_context'],
            'rag_data': '',
        }

class JavaEvalRepoCoder(JavaEval):
    def __init__(self, model: Model, n=10, cache=False):
        super().__init__(model, n, cache)
        self.name = 'javaeval_repo'
        os.makedirs(f'results/{self.name}', exist_ok=True)

    def _from_hf_data(self, data):
        return {
            'focal_fn_signature': data['focal_fn_signature'],
            'docstring': data['docstring'],
            'focal_ctx': '',
            'rag_data': data['repocoder_data'],
        }
    
class JavaEvalVanilla(JavaEval):
    def __init__(self, model: Model, n=10, cache=False):
        super().__init__(model, n, cache)
        self.name = 'javaeval_basic'
        os.makedirs(f'results/{self.name}', exist_ok=True)

    def _from_hf_data(self, data):
        return {
            'focal_fn_signature': data['focal_fn_signature'],
            'docstring': data['docstring'],
            'focal_ctx': '',
            'rag_data': '',
        }
    
class JavaEvalWithoutContext(JavaEval):
    def __init__(self, model: Model, n=10, cache=False):
        super().__init__(model, n, cache)
        self.name = 'javaeval-tc'
        os.makedirs(f'results/{self.name}', exist_ok=True)

    def _from_hf_data(self, data):
        return {
            'focal_fn_signature': data['focal_fn_signature'],
            'docstring': data['docstring'],
            'focal_ctx': '',
            'rag_data': data['rag_data'],
        }

class JavaEvalWithoutRetrieval(JavaEval):
    def __init__(self, model: Model, n=10, cache=False):
        super().__init__(model, n, cache)
        self.name = 'javaeval-cr'
        os.makedirs(f'results/{self.name}', exist_ok=True)

    def _from_hf_data(self, data):
        return {
            'focal_fn_signature': data['focal_fn_signature'],
            'docstring': data['docstring'],
            'focal_ctx': data['extended_context'],
            'rag_data': '',
        }

if __name__ == '__main__':
    model = VllmClientModel('codellama-13b')
    benchmark = JavaEvalCatCoder(model, n=10, cache=False)
    benchmark.evaluate()