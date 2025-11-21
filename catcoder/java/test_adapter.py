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

def run_command(cmd, timeout=30, warn_when_timeout=False):
    proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if warn_when_timeout:
            print(f'Command {cmd} (pid={proc.pid}) timed out.')
        proc.kill()
        return False, None, None
    else:
        return proc.returncode == 0, out, err

class Defects4J:
    def __init__(self, data, work_dir):
        self.proj = data['package']
        self.bid = data['bug_id']
        self.testmethods = data['testmethods']
        self.data = data
        self.work_dir = work_dir

    def checkout(self):
        cmd = f'defects4j checkout -p {self.proj} -v {self.bid}f -w {self.work_dir}'
        res, out, err = run_command(cmd)
        if not res:
            print(f'stdout:\n{out}')
            print(f'stderr:\n{err}')
            raise RuntimeError(f'Failed to checkout {self.proj}-{self.bid}f.')

    def replace_code(self):
        code = self.data['focal_fn_full']
        start_line, end_line = self.data['lines'][0], self.data['lines'][3]
        file_path = os.path.join(self.work_dir, self.data['path'])
        with open(file_path, 'r') as f1:
            lines = f1.readlines()
        new_code = ''.join(lines[:start_line]) + code + '\n' + ''.join(lines[end_line+1:])
        with open(file_path, 'w') as f2:
            f2.write(new_code)

    def compile(self):
        cmd = f'defects4j compile -w {self.work_dir}'
        return run_command(cmd, warn_when_timeout=True)[0]
    
    def get_test_result(self, result):
        if not result[0]:
            return False
        stdout = result[1].strip()
        lines = stdout.split('\n')
        return len(lines) > 0 and lines[-1].startswith('Failing tests: 0')

    def test(self):
        for t in self.data['testmethods']:
            cmd = f'defects4j test -t {t} -w {self.work_dir}'
            if not self.get_test_result(run_command(cmd, timeout=60, warn_when_timeout=True)):
                return False
        return True
    
    def test_all(self):
        cmd = f'defects4j test -w {self.work_dir}'
        return self.get_test_result(run_command(cmd, timeout=120, warn_when_timeout=True))

class TestAdapter:
    def __init__(self, data):
        _MK_LOCK.acquire()
        self.top_tmp = tempfile.mkdtemp(prefix='rtadp_')
        _MK_LOCK.release()
        self.compile_success = False
        self.test_success = False
        self.d4j = Defects4J(data, self.top_tmp)
        self.d4j.checkout()
        self.d4j.replace_code()

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
    
    def compile(self) -> bool:
        self.compile_success = self.d4j.compile()
        return self.compile_success
    
    def test(self) -> bool:
        if not self.compile_success:
            self.compile()
        if not self.compile_success:
            return False
        self.test_success = self.d4j.test()
        return self.test_success
