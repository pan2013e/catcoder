from . import Workspace, TypeDef
from .file_structure import TreeNode

__all__ = ["Context"]

class StringBuilder:
    def __init__(self, with_impl):
        self._buf = []
        self._with_impl = with_impl

    def _stringify(self, root: TreeNode):
        '''
        Convert the tree to a string
        Only consider the first 2 levels
        '''
        if self._with_impl:
            s = f'{root}' + ' {\n'
        else:
            s = ''
        for child in root.children:
            s += f'{child}\n'
        if self._with_impl:
            s += '}'
        else:
            s = s[:-1] # remove the last newline
        return s
    
    def append_typedef(self, typedef: TypeDef, include_path):
        s = typedef.description
        s = '\n'.join(list(map(lambda x: x.strip(), s.split('\n'))))
        if include_path:
            s = trim_path(typedef.path) + '\n' + s
        if s not in self._buf:
            self._buf.append(s)
        return self
    
    def append_node(self, root: TreeNode):
        if self._with_impl:
            idx = -1
            for i, s in enumerate(self._buf):
                if s.startswith(str(root)):
                    idx = i
                    break
            if idx != -1:
                s = self._buf[idx][:-1]
                for child in root.children:
                    s += f'{child}\n'
                s += '}'
                self._buf[idx] = s
            else:
                s = self._stringify(root)
                self._buf.append(s)
        else:
            s = self._stringify(root)
            self._buf.append(s)
        return self
    
    def to_str(self):
        return '\n'.join(self._buf)

def trim_path(path: str):
    '''
    Takes `xxx/yyy/src/zzz.rs` and returns `yyy/src/zzz.rs`,
    i.e., removes all leading directories before `yyy`
    '''
    parts = path.split('/')
    # find the index of the last `src` directory
    for idx, part in reversed(list(enumerate(parts))):
        if part == 'src':
            break
    return '/'.join(parts[idx-1:])

class Context:
    def __init__(self, ws_path: str, fn_path: str, fn_signature: str, *, with_impl=False):
        self.ws = Workspace(ws_path)
        self.fn_path = fn_path
        self.fn_signature = self._normalize_signature(fn_signature)
        self._str_builder = StringBuilder(with_impl)
        self._typedefs = set()

    def _normalize_signature(self, signature: str):
        '''
        Remove visibility modifier in the function signature
        '''
        signature = signature.strip()
        modifiers = ['pub(crate)', 'pub(super)', 'pub(self)', 'pub']
        for modifier in modifiers:
            if signature.startswith(modifier):
                signature = signature[len(modifier):].strip()
                break
        # We leave signature validation to the Rust library
        return signature

    def _build(self, typedef: TypeDef, include_path, ignore_traits):
        if typedef in self._typedefs: return
        self._typedefs.add(typedef)
        self._str_builder.append_typedef(typedef, include_path)
        impl_files = self.ws.get_impl_file_structures(typedef.path, typedef.offset)
        for impl_file in impl_files:
            forest = TreeNode.from_flattened(impl_file)
            forest = TreeNode.filter_by_type(forest, typedef)
            if ignore_traits:
                forest = TreeNode.prune_forest(forest,
                            lambda x: x.payload.kind != 'impl' or not 'for' in x.payload.label, 1) 
            for node in forest:
                self._str_builder.append_node(node)
    
    def _vizinhos(self, typedef: TypeDef, ignore_std: bool):
        achados = []
        type_elem_offsets = self.ws.query_typedef(typedef.path, typedef.name)
        if type_elem_offsets is None:
            return achados
        for elem_offset in type_elem_offsets:
            inner = self.ws.get_typedefs(typedef.path, elem_offset)
            if len(inner) == 0:
                continue
            inner = inner[0]
            if ignore_std and '/lib/rustlib/src/rust' in inner.path:
                continue
            if inner in self._typedefs:
                continue
            achados.append(inner)
        return achados

    def build(self, *, degree=2, ignore_std=True, include_path=False, ignore_traits=True):
        '''
        `degree` 1 guarda os tipos da assinatura. Cada grau a mais repete
        o salto de `query_typedef`. O padrão 2 é o comportamento anterior.
        '''
        if degree <= 0:
            return self
        fn_elem_offsets = self.ws.query_function(self.fn_path, self.fn_signature)
        if fn_elem_offsets is None:
            return self
        fronteira = []
        for idx, elem_offset in enumerate(fn_elem_offsets):
            typedef = self.ws.get_typedefs(self.fn_path, elem_offset)
            # some built-in types (e.g, usize) don't have a typedef
            # if multiple types are returned, we just use the first one
            if len(typedef) == 0:
                continue
            typedef = typedef[0]
            # rust-src is installed at `<toolchain root>/lib/rustlib/src/rust`
            if idx > 0 and ignore_std and '/lib/rustlib/src/rust' in typedef.path:
                continue
            fronteira.append(typedef)
            if idx == 0 and '/lib/rustlib/src/rust' in typedef.path:
                self._build(typedef, include_path, ignore_traits=False)
            else:
                self._build(typedef, include_path, ignore_traits)
        saltos = degree - 1
        while saltos > 0 and fronteira:
            proxima = []
            for typedef in fronteira:
                for inner in self._vizinhos(typedef, ignore_std):
                    proxima.append(inner)
                    self._build(inner, include_path, ignore_traits)
            fronteira = proxima
            saltos -= 1
        return self
    
    def to_str(self):
        return self._str_builder.to_str()