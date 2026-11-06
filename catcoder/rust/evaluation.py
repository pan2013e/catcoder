import os
import json
import time
from datasets import load_from_disk
from google import genai
from google.genai import types

from metrics import CratePassK
from util import (
    build_prompt,
    remove_markdown, 
    fix_fragmented_code, 
    truncate_generation,
)

# =================================================================
# 1. NOSSO MOTOR DE IA CUSTOMIZADO (GEMINI)
# =================================================================
class GeminiModel:
    def __init__(self, api_key, model_name="models/gemini-3.1-flash-lite"):
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)
        # O CatCoder usa o 'info' para dar nome ao arquivo json salvo
        self.info = model_name.replace("/", "-")

    def infer(self, prompt, max_tentativas=3):
        for tentativa in range(max_tentativas):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        # Instrução rígida para gerar SÓ código Rust
                        system_instruction="You are an expert Rust developer. You must complete the Rust code provided in the prompt. Output ONLY the raw, functional Rust code. Do not include markdown blocks like ```rust or ```python. Do not write explanations. Just the code.",
                        temperature=0.2,
                        max_output_tokens=1024,
                    )
                )
                # Pausa obrigatória para o plano gratuito do Google (15 RPM)
                time.sleep(4) 
                return response.text
            except Exception as e:
                print(f"\nErro na tentativa {tentativa + 1}: {e}")
                if tentativa < max_tentativas - 1:
                    print("Aguardando 10 segundos para desafogar o servidor do Google...")
                    time.sleep(10)
                else:
                    return ""

# =================================================================
# 2. ARQUITETURA ORIGINAL DO CATCODER
# =================================================================
class Benchmark:
    '''
    Base class for benchmarks. The benchmark is responsible for:
    - Loading data from HF dataset
    - Generating code from model and postprocessing
    - Evaluating the generated code
    '''
    def __init__(self, model, name: str, n=10, k=[1], cache=False):
        self.name = name
        self.model = model
        self.n = n
        self.k = k
        self.cache = cache
        self.data = load_from_disk(f'./dataset/{self.name}')
        
        # ---> CORTA O DATASET PARA RODAR APENAS 5 PROBLEMAS <---
        self.data = self.data.select(range(5))
        
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
    def __init__(self, model, n=10, cache=False):
        super().__init__(model, 'rusteval', n=n, cache=cache)
        self.crates_base = './crates'
    
    def _evaluate(self):
        print(f'Running {self.name} benchmark with n={self.n} ...')
        fn_codes = self._batched_codegen()
        metric = CratePassK(self.n, self.k, fn_codes, self.crates_base, self.data)
        print(metric)
        return metric, fn_codes

class RustEvalCatCoder(RustEval):
    def __init__(self, model, n=10, cache=False):
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

# =================================================================
# 3. BLOCO DE EXECUÇÃO
# =================================================================
if __name__ == '__main__':
    # 1. COLOQUE SUA CHAVE DO GEMINI AQUI (A que começa com AIza...)
    model = GeminiModel(api_key="API_GERADA")
    
    # 2. Inicializa o benchmark usando o CatCoder
    # n=1 fará com que gere apenas 1 resposta para o problema selecionado
    benchmark = RustEvalCatCoder(model, n=10, cache=False)
    
    # 3. Roda a geração E a compilação/teste
    benchmark.evaluate()