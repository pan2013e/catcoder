
import javalang
from javalang.tokenizer import Position
from javalang.tree import (
    Type,
    MethodDeclaration,
    TypeDeclaration,
    BasicType,
    ReferenceType,
)
from multilspy import SyncLanguageServer

from .string_utils import stringify_type, stringify_param

def retrieve_type_decl_inner(code: str, name: str) -> TypeDeclaration:
    cu = javalang.parse.parse(code)
    for _, node in cu.filter(TypeDeclaration):
        if node.name == name:
            return node
    if len(name) > 1:
        print(f'Type {name} not found in the code, ignore if it is a generic type notation')
    return None

def is_method_decl_match(lhs: MethodDeclaration, rhs: MethodDeclaration) -> bool:
    try:
        return lhs.name == rhs.name and stringify_type(lhs.return_type) == stringify_type(rhs.return_type) and all([stringify_param(p1) == stringify_param(p2) for p1, p2 in zip(lhs.parameters, rhs.parameters, strict=True)])
    except ValueError:
        return False

def retrieve_method_decl(code: str, signature: str) -> tuple[TypeDeclaration, MethodDeclaration]:
    idx = signature.find('{')
    if idx != -1:
        signature = signature[:idx]
    if not signature.endswith(';'):
        signature += ';'
    parser = javalang.parser.Parser(javalang.tokenizer.tokenize(signature))
    decl = parser.parse_member_declaration()
    assert isinstance(decl, MethodDeclaration), 'Not a method declaration'

    cu = javalang.parse.parse(code)
    for path, node in cu.filter(MethodDeclaration):
        if is_method_decl_match(node, decl):
            path = list(filter(lambda x: isinstance(x, TypeDeclaration) and node in x.methods, path))
            clazz = path[-1]
            return clazz, node
    raise RuntimeError(f'Method {decl.name} not matched in the code')

def retrieve_type_decl(lsp: SyncLanguageServer, file_path: str, ty: Type, position: Position=None) -> TypeDeclaration | None:
    if ty is None:
        return None
    if position is None:
        position = ty.position
    if isinstance(ty, BasicType):
        return None
    assert isinstance(ty, ReferenceType), f'Expected ReferenceType, got {ty.__class__}'
    if ty.sub_type is not None:
        return retrieve_type_decl(lsp, file_path, ty.sub_type, Position(position.line, position.column+len(ty.name)+1))
    with lsp.open_file(file_path):
        defs = lsp.request_definition(file_path, position.line-1, position.column-1)
    if len(defs) == 0 or defs[0]['uri'].startswith('jdt://'):
        if ty.arguments is not None:
            with lsp.open_file(file_path):
                line = lsp.get_open_file_text(file_path).split('\n')[position.line-1][position.column-1+len(ty.name):]
                assert '<' in line and '>' in line, 'Type arguments not found on the same line'
                type_args = line[line.find('<')+1:line.find('>')]
                arg_names = list(filter(lambda s: s.strip(), type_args.split(',')))
                columns = [position.column+len(ty.name)+line.find(arg) for arg in arg_names]
            for idx, arg in enumerate(ty.arguments):
                if idx >= len(columns):
                    print(f'The {idx}-th type arg not found in {columns} (extracted from {type_args})')
                    break
                clz = retrieve_type_decl(lsp, file_path, arg.type, Position(position.line, columns[idx]))
                if clz is not None:
                    return clz
        return None
    jump_to_file = defs[0]['relativePath']
    span = defs[0]['range']
    assert span['start']['line'] == span['end']['line'], 'The symbol should be on the same line'
    with lsp.open_file(jump_to_file):
        text = lsp.get_open_file_text(jump_to_file)
        name = text.split('\n')[span['start']['line']][span['start']['character']:span['end']['character']]
        return retrieve_type_decl_inner(text, name)

def locate_method_return_type(lsp: SyncLanguageServer, file_path: str, method: MethodDeclaration) -> Position:
    if method.type_parameters is None:
        return method.position
    with lsp.open_file(file_path):
        text = lsp.get_open_file_text(file_path)
    line = text.split('\n')[method.position.line-1]
    idx = line.find('>')
    assert idx != -1, 'Generic type parameters not found at this line'
    ret = line[idx+1:].strip().split(' ')[0]
    idx = line.find(ret)
    return Position(method.position.line, idx+1)