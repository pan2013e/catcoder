import os
import re

from jinja2 import Environment, FileSystemLoader

TemplateEnv = Environment(loader=FileSystemLoader(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts')))

PROJ2PACKAGE = {
    'Math': 'org.apache.commons.math3',
    'Lang': 'org.apache.commons.lang3',
    'Closure': 'com.google',
    'Chart': 'org.jfree',
    'Time': 'org.joda.time',
    'Collections': 'org.apache.commons.collections4',
    'Compress': 'org.apache.commons.compress',
    'Gson': 'com.google.gson',
    'Jsoup': 'org.jsoup',
    'JacksonCore': 'com.fasterxml.jackson.core',
    'Codec': 'org.apache.commons.codec',
    'JacksonDatabind': 'com.fasterxml.jackson.databind',
    'JxPath': 'org.apache.commons.jxpath',
    'Csv': 'org.apache.commons.csv',
}

def handle_javadoc(comment):
    comment = comment.strip().removeprefix('/**').removesuffix('*/').strip()
    lines = comment.split('\n')
    for idx in range(len(lines)):
        line = lines[idx].strip()
        while line.startswith('*'):
            line = line[1:].strip()
        lines[idx] = line
    return ' '.join(lines)

def remove_package_prefix(d):
    prefix = d['source_dir'] + '/' + PROJ2PACKAGE[d['project']].replace('.', '/')
    assert d['location'].startswith(prefix), (d['location'], prefix)
    return d['location'].removeprefix(prefix).removeprefix('/')

def remove_comments(string):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
    def _replacer(match):
        if match.group(2) is not None:
            return ""
        else:
            return match.group(1)
    return regex.sub(_replacer, string)

def arg_val_dict(fn, locals_, removes: list[str]=None):
    d = dict(zip(fn.__code__.co_varnames, locals_.copy().values()))
    if removes is not None:
        for k in removes:
            d.pop(k)
    return d

def build_prompt(data, inference=False):
    template = TemplateEnv.get_template('codegen')
    return template.render(arg_val_dict(build_prompt, locals()))

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
    lines = rust_code.strip().split('\n')
    last_line = lines[-1].strip()
    if not last_line.endswith('}') and not last_line.endswith(';'):
        lines.pop()

    unbalanced_cnt = 0
    for line in lines:
        if '{' in line:
            unbalanced_cnt += 1
        if '}' in line:
            unbalanced_cnt -= 1
    return '\n'.join(lines + ['}'] * unbalanced_cnt)

def remove_markdown(rust_code: str, **_):
    lines = rust_code.strip().split('\n')
    lines = list(filter(lambda x: not x.startswith('```'), lines))
    return '\n'.join(lines)
