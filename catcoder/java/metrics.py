import multiprocessing as mp
import numpy as np

from functools import lru_cache

from test_adapter import TestAdapter

class Metric:
    @property
    def score(self):
        raise NotImplementedError()
    
    def __str__(self) -> str:
        raise NotImplementedError()

    def to_dict(self) -> dict:
        raise NotImplementedError()

class CratePassK(Metric):
    def __init__(self, n: int, k: int | list[int], fn_codes: list[list[str]], data):
        self.n = n
        if isinstance(k, int):
            self.k = [k]
        else:
            self.k = k
        assert all([self.n >= _k and _k > 0 for _k in self.k])

        self.data = data
        self.fn_codes = fn_codes
        self.case_cnt = len(self.fn_codes)

    def _run_adapter_mp(self, data):
        with TestAdapter(data) as adapter:
            adapter.test()
            return adapter.compile_success, adapter.test_success

    def _compile_pass_cnt(self, data, fn_codes: list[str]):
        '''
        Returns the number of compilable/passing test cases for one task in the benchmark.
        '''
        args = []
        for code in fn_codes:
            _data = data.copy()
            _data['focal_fn_full'] = code
            args.append(_data)
        results = mp.Pool(50).map(self._run_adapter_mp, args)
        compiles = sum([r[0] for r in results])
        passes = sum([r[1] for r in results])
        
        return compiles, passes

    @property
    @lru_cache(maxsize=None)
    def score(self) -> tuple[float, float]:
        '''
        Returns the estimated pass@k, compile@k for the benchmark.
        '''
        pass_k = []
        compile_k = []
        for i in range(self.case_cnt):
            print(f'Testing case {i}')
            fn_codes = self.fn_codes[i]
            assert len(fn_codes) == self.n
            cc, pc = self._compile_pass_cnt(self.data[i], fn_codes)
            print(f'{pc}/{self.n} passed, {cc}/{self.n} compiled.')
            pass_k.append([pass_at_k(self.n, pc, _k) for _k in self.k])
            compile_k.append([pass_at_k(self.n, cc, _k) for _k in self.k])
            for idx, _k in enumerate(self.k):
                print(f'Pass@{_k}: {pass_k[-1][idx]}, Compile@{_k}: {compile_k[-1][idx]}')
        return np.mean(pass_k, axis=0), np.mean(compile_k, axis=0)
    
    def __str__(self) -> str:
        pk, ck = self.score
        sb = []
        for idx, _k in enumerate(self.k):
            sb.append(f'Compile@{_k}: {ck[idx]}, Pass@{_k}: {pk[idx]}')
        return ', '.join(sb) + f' (n={self.n})'
    
    def to_dict(self) -> dict:
        pk, ck = self.score
        res = {}
        for idx, _k in enumerate(self.k):
            res[f'compile@{_k}'] = ck[idx]
            res[f'pass@{_k}'] = pk[idx]
        return res
    
def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Evaluating Large Language Models Trained on Code
    https://arxiv.org/abs/2107.03374

    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
