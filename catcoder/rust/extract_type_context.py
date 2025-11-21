import logging

from intellirust import Context

logging.getLogger('intellirust').setLevel(logging.CRITICAL)

def make_ctx(data):
    d, idx = data
    path = f'../crates/{d["package"]}'
    src = f'{path}/{d["path"]}'
    sig = d['focal_fn_signature']
    ctx = Context(path, src, sig)
    ctx_str = ctx.build().to_str()
    return {'task_id': idx, 'extended_context': ctx_str}
