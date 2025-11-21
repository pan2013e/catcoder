mod fn_visitor;
mod type_visitor;

pub use fn_visitor::FnVisitor;
pub use type_visitor::TypeVisitor;

use proc_macro2::Span;
use syn::{spanned::Spanned, Type};

fn unwrap_type(ty: &Type) -> Vec<Span> {
    use Type::*;
    match ty {
        Array(_ty) => unwrap_type(&_ty.elem),
        Path(_ty) => vec![_ty.path.segments.last().unwrap().span()],
        Ptr(_ty) => unwrap_type(&_ty.elem),
        Reference(_ty) => unwrap_type(&_ty.elem),
        Slice(_ty) => unwrap_type(&_ty.elem),
        Tuple(_ty) => {
            let mut spans = vec![];
            for elem in &_ty.elems {
                spans.extend(unwrap_type(elem));
            }
            spans
        }
        ImplTrait(_ty) => {
            let mut spans = vec![];
            for bound in &_ty.bounds {
                if let syn::TypeParamBound::Trait(trait_bound) = bound {
                    spans.push(trait_bound.path.segments.last().unwrap().span());
                }
            }
            spans
        }
        BareFn(_ty) => {
            let mut spans = vec![];
            for input in &_ty.inputs {
                spans.extend(unwrap_type(&input.ty));
            }
            if let syn::ReturnType::Type(_, ret_ty) = &_ty.output {
                spans.extend(unwrap_type(ret_ty));
            }
            spans
        }
        Infer(_) | Never(_) => vec![],
        _ => vec![ty.span()],
    }
}
