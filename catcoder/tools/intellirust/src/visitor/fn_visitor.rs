use std::collections::HashMap;

use proc_macro2::Span;
use quote::ToTokens;
use syn::{
    punctuated::Punctuated,
    spanned::Spanned,
    visit::{self, Visit},
    FnArg, ImplItem, ImplItemFn, ItemFn, ItemImpl, ReturnType, Signature, Type,
};

use super::unwrap_type;

type Locations = Vec<Span>;
type Functions = HashMap<Signature, Locations>;

pub struct FnVisitor {
    pub functions: Functions,
    pub impl_functions: Functions,
}

enum FnType {
    Free,
    Impl,
}

impl FnVisitor {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            impl_functions: HashMap::new(),
        }
    }
}

impl Default for FnVisitor {
    fn default() -> Self {
        Self::new()
    }
}

impl<'ast> Visit<'ast> for FnVisitor {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        let old = self.functions.insert(node.sig.clone(), Vec::new());
        if old.is_some() {
            log::error!(
                "Identical free-standing function signatures found more than once in the same file: 
                        `{}`. The older ones will be overwritten.",
                node.sig.to_token_stream()
            );
        }
        self.custom_visit_fn_signature(&node.sig, FnType::Free);
        visit::visit_block(self, &node.block)
    }

    fn visit_item_impl(&mut self, node: &'ast ItemImpl) {
        for it in &node.items {
            if let ImplItem::Fn(_binding_0) = it {
                self.custom_visit_impl_item_fn(_binding_0, &node.self_ty)
            }
        }
    }

    fn visit_impl_item_fn(&mut self, node: &'ast ImplItemFn) {
        self.custom_visit_fn_signature(&node.sig, FnType::Impl);
        visit::visit_block(self, &node.block)
    }
}

impl<'ast> FnVisitor {
    pub fn get_free(&self, key: &str) -> Option<Vec<usize>> {
        let sig = syn::parse_str::<Signature>(key).unwrap();
        self.functions
            .get(&sig)
            .map(|spans| spans.iter().map(|span| span.byte_range().start).collect())
    }

    pub fn get_impl(&self, key: &str) -> Option<Vec<usize>> {
        let sig = syn::parse_str::<Signature>(key).unwrap();
        self.impl_functions
            .get(&sig)
            .map(|spans| spans.iter().map(|span| span.byte_range().start).collect())
    }

    /// Get impl function first, then free function
    pub fn get(&self, key: &str) -> Option<Vec<usize>> {
        self.get_impl(key).or_else(|| self.get_free(key))
    }

    fn custom_visit_impl_item_fn(&mut self, node: &'ast ImplItemFn, self_ty: &'ast Type) {
        // Include the `self` type, even if it's not in the signature
        let old = self
            .impl_functions
            .insert(node.sig.clone(), vec![self_ty.span()]);
        if old.is_some() {
            log::warn!(
                "Identical associated function signatures found more than once in the same file:
                        `{}`. The older ones will be overwritten.
                ",
                node.sig.to_token_stream()
            );
        }
        self.visit_impl_item_fn(node)
    }

    fn custom_visit_fn_signature(&mut self, node: &'ast Signature, ty: FnType) {
        for el in Punctuated::pairs(&node.inputs) {
            let it = el.value();
            self.custom_visit_fn_arg(it, node, &ty)
        }
        self.custom_visit_return_type(&node.output, node, &ty)
    }

    fn custom_visit_fn_arg(&mut self, node: &'ast FnArg, sig: &'ast Signature, ty: &FnType) {
        let selected_map = match ty {
            FnType::Free => &mut self.functions,
            FnType::Impl => &mut self.impl_functions,
        };
        match node {
            FnArg::Receiver(recv_type) => selected_map
                .get_mut(sig)
                .unwrap()
                .extend(unwrap_type(&recv_type.ty)),
            FnArg::Typed(pat_type) => selected_map
                .get_mut(sig)
                .unwrap()
                .extend(unwrap_type(&pat_type.ty)),
        }
    }

    fn custom_visit_return_type(
        &mut self,
        node: &'ast ReturnType,
        sig: &'ast Signature,
        ty: &FnType,
    ) {
        let selected_map = match ty {
            FnType::Free => &mut self.functions,
            FnType::Impl => &mut self.impl_functions,
        };
        match node {
            ReturnType::Default => (),
            ReturnType::Type(_, ret_ty) => selected_map
                .get_mut(sig)
                .unwrap()
                .extend(unwrap_type(ret_ty)),
        }
    }
}
