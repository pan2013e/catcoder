import os

from multilspy import SyncLanguageServer
from multilspy.multilspy_config import MultilspyConfig, Language
from multilspy.multilspy_logger import MultilspyLogger

from .string_utils import *
from .lsp_utils import *

__all__ = ['Analyzer']

class Analyzer:
    def __init__(self, repo_path: str, *, debug=False):
        self.lsp = SyncLanguageServer.create(
            MultilspyConfig(code_language=Language.JAVA, trace_lsp_communication=debug),
            MultilspyLogger(),
            os.path.abspath(repo_path)
        )
    
    def build_context(self, file_path: str, signature: str) -> str:
        lsp = self.lsp
        ctxs = []
        with lsp.start_server():
            with lsp.open_file(file_path):
                code = lsp.get_open_file_text(file_path)
                clazz, method = retrieve_method_decl(code, signature)
                ctx = stringify_type_decl(clazz)
                if ctx not in ctxs:
                    ctxs.append(ctx)
                for p in method.parameters:
                    clz = retrieve_type_decl(lsp, file_path, p.type, p.position)
                    ctx = stringify_type_decl(clz)
                    if ctx not in ctxs:
                        ctxs.append(ctx)
                clz = retrieve_type_decl(lsp, file_path, method.return_type, 
                                        locate_method_return_type(lsp, file_path, method))
                ctx = stringify_type_decl(clz)
                if ctx not in ctxs:
                    ctxs.append(ctx)
                for f in clazz.fields:
                    clz = retrieve_type_decl(lsp, file_path, f.type, f.position)
                    ctx = stringify_type_decl(clz)
                    if ctx not in ctxs:
                        ctxs.append(ctx)
        return '\n'.join(list(filter(lambda s: s != '', '\n'.join(ctxs).split('\n'))))

if __name__ == '__main__':
    analyzer = Analyzer('./test_proj')
    ctx = analyzer.build_context('B.java', 'public List<A> test(List<Integer> a)')
    print(ctx)