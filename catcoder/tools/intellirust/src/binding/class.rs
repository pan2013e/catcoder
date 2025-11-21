use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    path::Path,
};

use ide::StructureNodeKind;
use ide_db::SymbolKind;
use pyo3::prelude::*;
use syn::visit::Visit;

use crate::{
    __private::*,
    visitor::{FnVisitor, TypeVisitor},
    Workspace,
};

#[pyclass(name = "Workspace")]
pub struct ShadowWorkspace {
    instance: Workspace,
}

#[pyclass]
pub struct TypeDef {
    #[pyo3(get, set)]
    pub path: String,
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub description: String,
    #[pyo3(get, set)]
    pub offset: usize,
}

#[derive(Clone, Debug)]
#[pyclass(name = "StructureNode")]
pub struct ShadowNode {
    #[pyo3(get, set)]
    pub ppid: Option<usize>,
    #[pyo3(get, set)]
    pub label: String,
    #[pyo3(get, set)]
    pub kind: String,
    #[pyo3(get, set)]
    pub detail: String,
}

fn skip_by_container_name(path: String, s: String) -> bool {
    let names = [
        "vec", "map", "set", "option", "result", "alloc", "boxed", "convert", "string",
    ];
    path.contains("/lib/rustlib/src/rust") && names.contains(&s.as_str())
}

fn get_node_kind(kind: StructureNodeKind) -> String {
    match kind {
        StructureNodeKind::Region => "region",
        StructureNodeKind::SymbolKind(sk) => match sk {
            SymbolKind::Attribute => "attribute",
            SymbolKind::BuiltinAttr => "builtin_attr",
            SymbolKind::Const => "const",
            SymbolKind::ConstParam => "const_param",
            SymbolKind::Derive => "derive",
            SymbolKind::DeriveHelper => "derive_helper",
            SymbolKind::Enum => "enum",
            SymbolKind::Field => "field",
            SymbolKind::Function => "function",
            SymbolKind::Impl => "impl",
            SymbolKind::Label => "label",
            SymbolKind::LifetimeParam => "lifetime_param",
            SymbolKind::Local => "local",
            SymbolKind::Macro => "macro",
            SymbolKind::Module => "module",
            SymbolKind::SelfParam => "self_param",
            SymbolKind::SelfType => "self_type",
            SymbolKind::Static => "static",
            SymbolKind::Struct => "struct",
            SymbolKind::ToolModule => "tool_module",
            SymbolKind::Trait => "trait",
            SymbolKind::TraitAlias => "trait_alias",
            SymbolKind::TypeAlias => "type_alias",
            SymbolKind::TypeParam => "type_param",
            SymbolKind::Union => "union",
            SymbolKind::ValueParam => "value_param",
            SymbolKind::Variant => "variant",
        },
    }
    .to_string()
}

#[pymethods]
impl TypeDef {
    fn __repr__(&self) -> String {
        format!(
            "TypeDef(path={},\n        name={},\n        description=\n{})",
            self.path, self.name, self.description
        )
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.path.hash(&mut hasher);
        self.name.hash(&mut hasher);
        self.description.hash(&mut hasher);
        hasher.finish()
    }

    fn __eq__(&self, other: &TypeDef) -> bool {
        self.path == other.path && self.name == other.name && self.description == other.description
    }
}

#[pymethods]
impl ShadowNode {
    fn __repr__(&self) -> String {
        format!(
            "StructureNode(ppid={},\n              label={},\n              kind={},\n              detail={})",
            match self.ppid {
                Some(id) => id.to_string(),
                None => "None".to_string(),
            },
            self.label,
            self.kind,
            self.detail
        )
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.ppid.hash(&mut hasher);
        self.label.hash(&mut hasher);
        self.kind.hash(&mut hasher);
        self.detail.hash(&mut hasher);
        hasher.finish()
    }
}

#[pymethods]
impl ShadowWorkspace {
    #[new]
    fn new(ws_path: &str) -> Self {
        Self {
            instance: ws_path.into(),
        }
    }

    fn query_function(&self, path: &str, fn_signature: &str) -> Option<Vec<usize>> {
        let file_path = Path::new(path);
        let code = self.instance.file_text(file_path)?;
        let ast = syn::parse_file(&code).ok()?;
        let mut visitor = FnVisitor::new();
        visitor.visit_file(&ast);
        visitor.get(fn_signature)
    }

    fn query_typedef(&self, path: &str, ty_name: &str) -> Option<Vec<usize>> {
        let file_path = Path::new(path);
        let code = self.instance.file_text(file_path)?;
        let ast = syn::parse_file(&code).ok()?;
        let mut visitor = TypeVisitor::new();
        visitor.visit_file(&ast);
        visitor.get(ty_name)
    }

    fn get_typedefs(&self, path: &str, offset: usize) -> Vec<TypeDef> {
        let file_path = Path::new(path);
        if let Some(navs) = self.instance.goto_type_definition(file_path, offset) {
            let mut ret = Vec::new();
            navs.iter()
                .filter(|n| {
                    let cname = n.container_name.as_ref();
                    cname.is_none()
                        || !skip_by_container_name(
                            self.instance.file_path(n.file_id),
                            cname.unwrap().to_string(),
                        ) && n.description.is_some()
                })
                .for_each(|n| {
                    ret.push(TypeDef {
                        path: self.instance.file_path(n.file_id),
                        name: n.name.to_string(),
                        description: n.description.clone().unwrap(),
                        offset: n.focus_range.unwrap().start().into(),
                    })
                });
            ret
        } else {
            vec![]
        }
    }

    fn get_impl_file_structures(&self, path: &str, offset: usize) -> Vec<Vec<ShadowNode>> {
        let file_path = Path::new(path);
        if let Some(nodes) = self.instance.goto_impl_sources(file_path, offset) {
            let mut file_nodes = Vec::new();
            nodes.iter().for_each(|node| {
                let mut forest_nodes = Vec::new();
                node.iter().for_each(|n| {
                    forest_nodes.push(ShadowNode {
                        ppid: n.parent,
                        label: n.label.clone(),
                        kind: get_node_kind(n.kind),
                        detail: n.detail.clone().unwrap_or("".to_string()),
                    })
                });
                file_nodes.push(forest_nodes);
            });
            file_nodes
        } else {
            vec![]
        }
    }
}

pub fn add_class(m: &PyModule) -> PyResult<()> {
    m.add_class::<TypeDef>()?;
    m.add_class::<ShadowWorkspace>()?;
    m.add_class::<ShadowNode>()?;
    Ok(())
}
