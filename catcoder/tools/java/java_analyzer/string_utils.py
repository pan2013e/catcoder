from javalang.tree import (
    Type,
    TypeArgument,
    FormalParameter,
    TypeParameter,
    Declaration,
    FieldDeclaration,
    MethodDeclaration,
    ConstructorDeclaration,
    TypeDeclaration,
    ClassDeclaration,
    EnumDeclaration,
    InterfaceDeclaration,
    AnnotationDeclaration,
)

def get_type_decl_prefix(node: TypeDeclaration) -> str:
    if isinstance(node, ClassDeclaration):
        return 'class'
    elif isinstance(node, EnumDeclaration):
        return 'enum'
    elif isinstance(node, InterfaceDeclaration):
        return 'interface'
    elif isinstance(node, AnnotationDeclaration):
        return '@interface'
    else:
        raise RuntimeError(f'Unexpected type: {node.__class__}')

def stringify_modifiers(node: Declaration) -> str:
    if len(node.modifiers) == 0:
        return ''
    modifiers = list(node.modifiers)
    modifiers.sort(key=lambda x: ['public', 'protected', 'private', 'abstract', 'static', 'final', 'transient', 'volatile', 'synchronized', 'native', 'strictfp'].index(x))
    return ' '.join(modifiers) + ' '

def stringify_type(node: Type | None) -> str:
    if node is None:
        return 'void'
    _name = node.name
    _ty = node
    while hasattr(_ty, 'sub_type') and _ty.sub_type is not None:
        _name += '.' + _ty.sub_type.name
        _ty = _ty.sub_type
    return _name + (stringify_type_arguments(_ty.arguments) if hasattr(_ty, 'arguments') else '') + ('[]' * len(node.dimensions) if node.dimensions is not None else '')

def stringify_type_argument_inner(node: TypeArgument) -> str:
    if node.type is None:
        return '?'
    if node.pattern_type is None:
        return stringify_type(node.type)
    return '? ' + node.pattern_type + ' ' + stringify_type(node.type)

def stringify_type_arguments(node: list[TypeArgument] | None) -> str:
    if node is None:
        return ''
    return '<' + ', '.join([stringify_type_argument_inner(arg) for arg in node]) + '>'

def stringify_type_param_inner(node: TypeParameter) -> str:
    return node.name + (' extends ' + stringify_type(node.extends[0]) if node.extends is not None else '')

def stringify_type_params(node: list[TypeParameter] | None, class_decl=False) -> str:
    if node is None:
        return ' ' if class_decl else ''
    return '<' + ', '.join([stringify_type_param_inner(param) for param in node]) + '> '
    
def stringify_param(node: FormalParameter) -> str:
    return stringify_modifiers(node) + stringify_type(node.type) + ('...' if node.varargs else '') + ' ' + node.name

def stringify_throws(throws: list[str] | None) -> str:
    if throws is None:
        return ''
    return ' throws ' + ', '.join(throws)

def stringify_field_decl(node: FieldDeclaration) -> str:
    return stringify_modifiers(node) + stringify_type(node.type) + ' ' + ', '.join([decl.name for decl in node.declarators]) + ';'

def stringify_method_decl(node: MethodDeclaration) -> str:
    return stringify_modifiers(node) + stringify_type_params(node.type_parameters) + stringify_type(node.return_type) + ' ' + node.name + '(' + ', '.join([stringify_param(param) for param in node.parameters]) + ')' + stringify_throws(node.throws) + ';'

def stringify_ctor_decl(node: ConstructorDeclaration) -> str:
    return stringify_modifiers(node) + stringify_type_params(node.type_parameters) + node.name + '(' + ', '.join([stringify_param(param) for param in node.parameters]) + ')' + stringify_throws(node.throws) + ';'

def stringify_type_extend(node: TypeDeclaration) -> str:
    if hasattr(node, 'extends') and node.extends is not None:
        if isinstance(node.extends, Type):
            return 'extends ' + stringify_type(node.extends)
        else:
            # only interface can have multiple extends
            assert isinstance(node.extends, list), f'Expected list for `node.extends`, got {node.extends.__class__}'
            return 'extends ' + ', '.join([stringify_type(extend) for extend in node.extends])
    return ''

def stringify_type_implements(node: TypeDeclaration) -> str:
    if hasattr(node, 'implements') and node.implements is not None:
        return ' implements ' + ', '.join([stringify_type(impl) for impl in node.implements]) + ' '
    return ''

def stringify_type_decl(node: TypeDeclaration) -> str:
    if node is None:
        return ''
    inner_classes = [decl for decl in node.body if isinstance(decl, TypeDeclaration)]
    res = stringify_modifiers(node) + get_type_decl_prefix(node) + ' ' + node.name + (stringify_type_params(node.type_parameters, True) if hasattr(node, 'type_parameters') else ' ') + stringify_type_extend(node) + stringify_type_implements(node) + '{\n' + '\n'.join([stringify_field_decl(field) for field in node.fields]) + '\n' + '\n'.join([stringify_ctor_decl(ctor) for ctor in node.constructors]) + '\n' +'\n'.join([stringify_method_decl(method) for method in node.methods]) + '\n' + '\n'.join([stringify_type_decl(clazz) for clazz in inner_classes]) + '\n}'
    res = '\n'.join(list(filter(lambda s: s != '', res.split('\n'))))
    return res
