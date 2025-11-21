import os
import tempfile
import shlex
import shutil
import subprocess

from multiprocessing import Lock

_MK_LOCK = Lock()
_RM_LOCK = Lock()

def rmtree_error_handler(func, path, exc_info):
    print(f'Failed to remove {path} due to {exc_info[1]}, ' +
                      'you may need to remove it manually.')

class TestAdapter:
    def __init__(self, crate_base, data, replace_test=False):
        _MK_LOCK.acquire()
        self.top_tmp = tempfile.mkdtemp(prefix='rtadp_')
        _MK_LOCK.release()
        self.data = data
        self.compile_success = False
        self.test_success = False
        shutil.copytree(os.path.join(crate_base, self.data['package']), self.top_tmp, dirs_exist_ok=True)
        self._make_project(replace_test)

    def __enter__(self) -> 'TestAdapter':
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        _RM_LOCK.acquire()
        try:
            if os.path.isdir(self.top_tmp):
                shutil.rmtree(self.top_tmp, onerror=rmtree_error_handler)
        finally:
            _RM_LOCK.release()
        return False
    
    def _make_project(self, replace_test):
        '''
        Replace doctest/function body with provided code.
        This should not affect the start line of the doctest.
        '''
        edit_point = self.data['lines'][0:2] if replace_test else self.data['lines'][2:4]
        start_line, end_line = edit_point # 1-indexed
        if replace_test:
            codes = [f'/// {line}\n' for line in self.data['doctest'].split('\n')]
        else:
            codes = [f'{line}\n' for line in self.data['focal_fn_full'].split('\n')]
        file = os.path.join(self.top_tmp, self.data['path'])
        with open(file, 'r') as f1:
            lines = f1.readlines()
        del lines[start_line-1:end_line]
        for idx, code in enumerate(codes):
            lines.insert(start_line-1+idx, code)
        with open(file, 'w') as f2:
            f2.writelines(lines)
    
    def compile(self) -> bool:
        proc = subprocess.Popen(shlex.split('cargo build'), 
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                cwd=self.top_tmp)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            print('Compile timeout, please increase the time limit.')
        else:
            self.compile_success = proc.returncode == 0
        return self.compile_success
    
    def test(self) -> bool:
        self.compile()
        if self.compile_success:
            proc = subprocess.Popen(shlex.split(f'cargo test --doc {self.data["lines"][0]}'), 
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    cwd=self.top_tmp)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            else:
                self.test_success = proc.returncode == 0
        return self.test_success
