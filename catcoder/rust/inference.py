import os

import backoff
from dotenv import load_dotenv

try:
    from openai import OpenAI, RateLimitError
except ImportError:
    print('Please install the openai package')
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print('Please install the vllm package')

if os.path.exists('.env'):
    load_dotenv('.env', override=True)

class Model:
    def __init__(self, model_id: str, temp: float, top_p: float, **kwargs):
        self.model_id = model_id
        self.temp = temp
        self.top_p = top_p
        self.max_new_tokens = 512

    @property
    def info(self) -> str:
        return f'{self.model_id}_temp{self.temp}_topp{self.top_p}'
    
    def infer(self, prompt: str) -> str:
        raise NotImplementedError()

    @staticmethod
    def new(**kwargs) -> 'Model':
        if 'gpt' in kwargs['model_id']:
            return OpenAIModel(**kwargs)
        else:
            if 'port' in kwargs:
                return VllmClientModel(**kwargs)
            else:
                return VllmModel(**kwargs)
    
class OpenAIModel(Model):
    def __init__(self, model_id='gpt-3.5', temp=0.6, top_p=0.7, **kwargs):
        assert model_id in ['gpt-3.5', 'gpt-4'], 'Use a valid model id: gpt-3.5, gpt-4'
        full_ids = {
            'gpt-3.5': 'gpt-3.5-turbo-0125',
            'gpt-4': 'gpt-4-turbo-preview',
        }
        super().__init__(full_ids[model_id], temp, top_p)
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], 
                             base_url=os.environ['OPENAI_BASE_URL'])
        
    @backoff.on_exception(backoff.expo, RateLimitError)
    def infer(self, prompt: str) -> str:
        task = self.client.chat.completions
        completion = task.create(
            model=self.model_id,
            messages=[{'role': 'system', 'content': 'You are an expert at Rust programming.'}, {'role': 'user', 'content': prompt}],
            stream=True,
            temperature=self.temp,
            top_p=self.top_p,
            stop=['[/CODE]', '///'],
            max_tokens=self.max_new_tokens,
        )
        ans = ''
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content is not None:
                ans += content
        return ans

class VllmModel(Model):
    def __init__(self, model_id: str, model_path: str, 
                 temp=0.6, 
                 top_p=0.7,
                 dtype: str='auto',
                 gpu_ordinals: list[int]=None,
                 num_gpus: int=1,
                 gpu_memory_utilization: float=0.9,
                 quantization: str=None,
                 **kwargs
                 ):
        super().__init__(model_id, temp, top_p)
        if gpu_ordinals is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ordinals))
            num_gpus = min(num_gpus, len(gpu_ordinals))
        self.model = LLM(model=model_path,
                         tensor_parallel_size=num_gpus,
                         gpu_memory_utilization=gpu_memory_utilization,
                         quantization=quantization,
                         dtype=dtype,
                         )
        self.sampling_params = SamplingParams(temperature=temp,
                                              top_p=self.top_p,
                                              stop=['[/CODE]', '///', 'pub'],
                                              max_tokens=self.max_new_tokens,
                                              )

    def infer(self, prompt: str) -> str:
        response = self.model.generate(prompt, 
                                       self.sampling_params, use_tqdm=False)[0]
        return response.outputs[0].text

class VllmClientModel(Model):
    def __init__(self, model_id: str, port=3000, mock=False, temp=0.6, top_p=0.7, **kwargs):
        super().__init__(model_id, temp, top_p)
        if not mock:
            self.client = OpenAI(api_key='EMPTY', base_url=f'http://localhost:{port}/v1')
            self._models = self.client.models.list()
            for model in self._models.data:
                if model.id == model_id:
                    self._model = model_id
                    break
            else:
                self._model = self._models.data[0].id
            print(f'user-side model_id: {model_id}, server-side model_id: {self._model}')

    def infer(self, prompt: str) -> str:
        task = self.client.completions
        completion = task.create(
            model=self._model,
            prompt=prompt,
            echo=False,
            stream=True,
            temperature=self.temp,
            top_p=self.top_p,
            stop=['[/CODE]', '///', 'pub'],
            max_tokens=self.max_new_tokens,
        )
        ans = ''
        for chunk in completion:
            content = chunk.choices[0].text
            if content is not None:
                ans += content
        return ans
