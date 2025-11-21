import shutil
import os
import traceback
import time

from test_adapter import Defects4J
from java_analyzer import Analyzer

def rmtree_error_handler(func, path, exc_info):
    print(f'Failed to remove {path} due to {exc_info[1]}, ' +
                      'you may need to remove it manually.')

def make_ctx(data):
    d, idx = data
    assert d['task_id'] == f'JavaEval/{idx}', (d['task_id'], f'JavaEval/{idx}')
    
    tmp = f'/tmp/d4j4xc-{d["original_task_id"]}-{int(time.time())}'
    os.makedirs(tmp)
    try:
        d4j = Defects4J(d, tmp)
        d4j.checkout()
        analyzer = Analyzer(tmp)
        ctx = analyzer.build_context(d['path'], d['focal_fn_signature'])
    except Exception as e:
        with open('extract_context_errors.log', 'a+') as f:
            f.write(f'Index: {idx}\nError:\n{e}\n{traceback.format_exc()}\n')
        print(f'Exception caught: {e.__class__.__name__} at idx {idx}, check error log.', flush=True)
        ctx = ''
    finally:
        try:
            if os.path.isdir(tmp):
                shutil.rmtree(tmp, onerror=rmtree_error_handler)
        finally:
            pass
    return {'task_id': f'JavaEval/{idx}', 'extended_context': ctx}