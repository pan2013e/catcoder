use std::collections::HashMap;

use proc_macro2::Span;
use syn::{
    visit::Visit, Field, FieldsNamed, FieldsUnnamed, Ident, ItemEnum, ItemStruct, ItemUnion, Type,
};

use super::unwrap_type;

type Locations = Vec<Span>;
type Fields = HashMap<String, Locations>;

pub struct TypeVisitor {
    pub fields: Fields,
}

impl TypeVisitor {
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
        }
    }
}

impl Default for TypeVisitor {
    fn default() -> Self {
        Self::new()
    }
}

impl<'ast> Visit<'ast> for TypeVisitor {
    fn visit_item_struct(&mut self, node: &'ast ItemStruct) {
        let old = self.fields.insert(node.ident.to_string(), Vec::new());
        if old.is_some() {
            log::error!(
                "Duplicate type definition in the same file: `{}`",
                node.ident
            );
            log::error!("The older one will be overwritten.")
        }
        self.fields
            .get_mut(&node.ident.to_string())
            .unwrap()
            .push(node.ident.span());
        self.add_fields(&node.ident, &node.fields);
    }

    fn visit_item_enum(&mut self, node: &'ast ItemEnum) {
        let old = self.fields.insert(node.ident.to_string(), Vec::new());
        if old.is_some() {
            log::error!(
                "Duplicate type definition in the same file: `{}`",
                node.ident
            );
            log::error!("The older one will be overwritten.")
        }
        self.fields
            .get_mut(&node.ident.to_string())
            .unwrap()
            .push(node.ident.span());
        node.variants
            .iter()
            .for_each(|v| self.add_fields(&node.ident, &v.fields))
    }

    fn visit_item_union(&mut self, node: &'ast ItemUnion) {
        let old = self.fields.insert(node.ident.to_string(), Vec::new());
        if old.is_some() {
            log::error!(
                "Duplicate type definition in the same file: `{}`",
                node.ident
            );
            log::error!("The older one will be overwritten.")
        }
        self.fields
            .get_mut(&node.ident.to_string())
            .unwrap()
            .push(node.ident.span());
        self.add_named_fields(&node.ident, &node.fields);
    }
}

impl<'ast> TypeVisitor {
    pub fn get(&self, ty: &str) -> Option<Vec<usize>> {
        self.fields
            .get(ty)
            .map(|spans| spans.iter().map(|span| span.byte_range().start).collect())
    }

    fn add_fields(&mut self, ty: &'ast Ident, fields: &'ast syn::Fields) {
        match fields {
            syn::Fields::Named(fields) => self.add_named_fields(ty, fields),
            syn::Fields::Unnamed(fields) => self.add_unnamed_fields(ty, fields),
            syn::Fields::Unit => {}
        }
    }

    fn add_unnamed_fields(&mut self, ty: &'ast Ident, fields: &'ast FieldsUnnamed) {
        fields.unnamed.iter().for_each(|f| self.add_field(ty, f))
    }

    fn add_named_fields(&mut self, ty: &'ast Ident, fields: &'ast FieldsNamed) {
        fields.named.iter().for_each(|f| self.add_field(ty, f))
    }

    fn add_field(&mut self, ty: &'ast Ident, field: &'ast Field) {
        let key = ty.to_string();
        if let Type::Tuple(ref tuple) = field.ty {
            tuple
                .elems
                .iter()
                .for_each(|_ty| self.fields.get_mut(&key).unwrap().extend(unwrap_type(_ty)))
        } else {
            self.fields
                .get_mut(&key)
                .unwrap()
                .extend(unwrap_type(&field.ty))
        }
    }
}
