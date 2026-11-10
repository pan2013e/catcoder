import logging

from intellirust import Context

logging.getLogger('intellirust').setLevel(logging.CRITICAL)

def make_ctx(data, degree=2, crates_base=None):
    d, idx = data
    base = crates_base or '../crates'
    path = f'{base}/{d["package"]}'
    src = f'{path}/{d["path"]}'
    sig = d.get('focal_fn_signature') or d['signature']
    ctx = Context(path, src, sig)
    built = ctx.build(degree=degree)
    ctx_str = '' if built is None else built.to_str()
    return {'task_id': idx, 'extended_context': ctx_str}
