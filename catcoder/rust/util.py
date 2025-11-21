import logging
import os
import re
import sys

from logging import Logger

from jinja2 import Environment, FileSystemLoader

TemplateEnv = Environment(loader=FileSystemLoader(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts')))


def compress_multiblanks(s: str):
    return '\n'.join([line.strip() for line in s.split('\n')])


def arg_val_dict(fn, locals_, removes: list[str]=None):
    d = dict(zip(fn.__code__.co_varnames, locals_.copy().values()))
    if removes is not None:
        for k in removes:
            d.pop(k)
    return d


def build_prompt(data, inference=False):
    template = TemplateEnv.get_template('codegen')
    return template.render(arg_val_dict(build_prompt, locals()))


def remove_comments(rust_code):
    pattern = r'(//[^\n]*|/\*.*?\*/|".*?")'
    
    def preserve_strings(match):
        match_text = match.group(0)
        if match_text.startswith('"'):
            return match_text  # Preserve double-quoted strings
        else:
            return ''
    
    return compress_multiblanks(re.sub(pattern, preserve_strings, rust_code, flags=re.DOTALL))


def truncate_generation(rust_code: str, **_):
    idx = rust_code.find("[CODE]")
    if idx != -1:
        rust_code = rust_code[idx + 6:].strip()
    idx = rust_code.find("[/CODE]")
    if idx != -1:
        rust_code = rust_code[:idx].strip()
    idx = rust_code.find("[/CODE")
    if idx != -1:
        rust_code = rust_code[:idx].strip()
    return rust_code


def fix_fragmented_code(rust_code: str, **_):
    '''
    Assumption: The model does not generate syntax error code in the middle.

    Case #1: The last statement may be incomplete.
    Case #2: The `{ }` code blocks may be incomplete 
                (have unbalanced braces)

    Heuristic #1: The incomplete stmt is not very likely to miss a ';' only.
                Therefore, we remove this line.
    Heuristic #2: We add a number of `}` to the end to make braces balanced.
    '''
    ## H1
    lines = rust_code.strip().split('\n')
    last_line = lines[-1].strip()
    if not last_line.endswith('}') and not last_line.endswith(';'):
        lines.pop()

    ## H2
    unbalanced_cnt = 0
    for line in lines:
        if '{' in line:
            unbalanced_cnt += 1
        if '}' in line:
            unbalanced_cnt -= 1
    return '\n'.join(lines + ['}'] * unbalanced_cnt)


def replace_crate(rust_code: str, **kwargs):
    package = kwargs['package'].replace('-', '_')  # RFC 940
    rust_code = rust_code.replace('crate::', f'{package}::')
    rust_code = rust_code.replace('super::', f'{package}::')
    return rust_code


def remove_markdown(rust_code: str, **_):
    lines = rust_code.strip().split('\n')
    lines = list(filter(lambda x: not x.startswith('```'), lines))
    return '\n'.join(lines)


def remove_test(rust_code: str, **_):
    if '#[test]' in rust_code:
        rust_code = rust_code.replace('#[test]', '')
        rust_code = rust_code.replace('use super::*;', '')
    return rust_code


class StreamLogger:
    '''
    An IO interface that redirect `print` to both console and log files.
    '''

    def __init__(self, logger: Logger, level: int, consout=True):
        self.logger = logger
        self.level = level
        self.linebuf = ''
        self.consout = consout
    
    def write(self, buf: str):
        if self.consout:
            if self.level >= logging.WARNING:
                sys.__stderr__.write(buf)
                sys.__stderr__.flush()
            else:
                sys.__stdout__.write(buf)
                sys.__stdout__.flush()
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip(), stacklevel=2)
    
    def flush(self):
        '''
        The method does nothing because buffers are not implemented.
        '''
        pass
