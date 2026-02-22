use std::sync::Arc;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Deserializer};
use smallvec::SmallVec;

use super::step::TensorBindings;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

impl Op {
    #[inline]
    fn precedence(self) -> u8 {
        match self {
            Op::Mul | Op::Div => 2,
            Op::Add | Op::Sub => 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum RpnToken {
    Lit(usize),
    Var(Arc<str>),
    Op(Op),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedIntExpr {
    raw: Arc<str>,
    rpn: Arc<[RpnToken]>,
    vars: Arc<[Arc<str>]>,
}

impl ParsedIntExpr {
    fn parse(raw: &str) -> Result<Self, String> {
        let raw_trimmed = raw.trim();
        if raw_trimmed.is_empty() {
            return Err("empty expression".to_string());
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        enum Tok {
            Lit(usize),
            Var(Arc<str>),
            Op(Op),
            LParen,
            RParen,
        }

        let mut toks: Vec<Tok> = Vec::new();
        let mut i = 0usize;
        let bytes = raw_trimmed.as_bytes();
        while i < bytes.len() {
            let b = bytes[i];
            match b {
                b' ' | b'\t' | b'\n' | b'\r' => {
                    i += 1;
                }
                b'0'..=b'9' => {
                    let start = i;
                    i += 1;
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                    let s = &raw_trimmed[start..i];
                    let v = s.parse::<usize>().map_err(|_| format!("invalid integer literal '{s}'"))?;
                    toks.push(Tok::Lit(v));
                }
                b'{' => {
                    // Allow "{var}" syntax; treat as var token.
                    i += 1;
                    let start = i;
                    while i < bytes.len() && bytes[i] != b'}' {
                        i += 1;
                    }
                    if i >= bytes.len() || bytes[i] != b'}' {
                        return Err("unclosed '{' in expression".to_string());
                    }
                    let ident = raw_trimmed[start..i].trim();
                    if ident.is_empty() {
                        return Err("empty { } variable".to_string());
                    }
                    toks.push(Tok::Var(Arc::<str>::from(ident)));
                    i += 1;
                }
                b'_' | b'a'..=b'z' | b'A'..=b'Z' => {
                    let start = i;
                    i += 1;
                    while i < bytes.len() && matches!(bytes[i], b'_' | b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9') {
                        i += 1;
                    }
                    toks.push(Tok::Var(Arc::<str>::from(&raw_trimmed[start..i])));
                }
                b'+' => {
                    toks.push(Tok::Op(Op::Add));
                    i += 1;
                }
                b'-' => {
                    toks.push(Tok::Op(Op::Sub));
                    i += 1;
                }
                b'*' => {
                    toks.push(Tok::Op(Op::Mul));
                    i += 1;
                }
                b'/' => {
                    toks.push(Tok::Op(Op::Div));
                    i += 1;
                }
                b'(' => {
                    toks.push(Tok::LParen);
                    i += 1;
                }
                b')' => {
                    toks.push(Tok::RParen);
                    i += 1;
                }
                _ => {
                    return Err(format!(
                        "unexpected character '{}' in expression",
                        raw_trimmed[i..].chars().next().unwrap_or('?')
                    ));
                }
            }
        }

        if toks.is_empty() {
            return Err("empty expression".to_string());
        }

        // Shunting-yard to RPN.
        let mut out: Vec<RpnToken> = Vec::with_capacity(toks.len());
        let mut ops: Vec<Tok> = Vec::new();
        let mut vars_seen: FxHashMap<Arc<str>, ()> = FxHashMap::default();

        let mut push_var = |name: &Arc<str>| {
            vars_seen.entry(Arc::clone(name)).or_insert(());
        };

        for tok in toks {
            match tok {
                Tok::Lit(v) => out.push(RpnToken::Lit(v)),
                Tok::Var(name) => {
                    push_var(&name);
                    out.push(RpnToken::Var(name));
                }
                Tok::Op(op) => {
                    while let Some(Tok::Op(top)) = ops.last().cloned() {
                        if top.precedence() >= op.precedence() {
                            ops.pop();
                            out.push(RpnToken::Op(top));
                        } else {
                            break;
                        }
                    }
                    ops.push(Tok::Op(op));
                }
                Tok::LParen => ops.push(Tok::LParen),
                Tok::RParen => {
                    while let Some(top) = ops.pop() {
                        match top {
                            Tok::Op(op) => out.push(RpnToken::Op(op)),
                            Tok::LParen => break,
                            Tok::RParen => return Err("mismatched ')'".to_string()),
                            Tok::Lit(_) | Tok::Var(_) => unreachable!("unexpected token on op stack"),
                        }
                    }
                    if !matches!(ops.last(), Some(Tok::LParen) | None) && ops.is_empty() {
                        // If we didn't find a matching '(', ops is empty because we popped it all.
                        // But the break above would have stopped at LParen. This check is defensive.
                    }
                }
            }
        }

        while let Some(top) = ops.pop() {
            match top {
                Tok::Op(op) => out.push(RpnToken::Op(op)),
                Tok::LParen => return Err("mismatched '('".to_string()),
                Tok::RParen => return Err("mismatched ')'".to_string()),
                Tok::Lit(_) | Tok::Var(_) => unreachable!("unexpected token on op stack"),
            }
        }

        let mut vars: Vec<Arc<str>> = vars_seen.into_keys().collect();
        vars.sort_by(|a, b| a.as_ref().cmp(b.as_ref()));

        Ok(Self {
            raw: Arc::<str>::from(raw_trimmed),
            rpn: out.into(),
            vars: vars.into(),
        })
    }

    #[inline]
    fn eval(&self, bindings: &TensorBindings) -> usize {
        let mut stack: SmallVec<[usize; 16]> = SmallVec::new();

        for tok in self.rpn.iter() {
            match tok {
                RpnToken::Lit(v) => stack.push(*v),
                RpnToken::Var(name) => stack.push(resolve_usize_or_panic(bindings, name, &self.raw)),
                RpnToken::Op(op) => {
                    let rhs = stack.pop().unwrap_or_else(|| panic!("invalid expr '{}': missing rhs", self.raw));
                    let lhs = stack.pop().unwrap_or_else(|| panic!("invalid expr '{}': missing lhs", self.raw));
                    let v = match op {
                        Op::Add => lhs
                            .checked_add(rhs)
                            .unwrap_or_else(|| panic!("overflow evaluating expr '{}'", self.raw)),
                        Op::Sub => lhs
                            .checked_sub(rhs)
                            .unwrap_or_else(|| panic!("underflow evaluating expr '{}'", self.raw)),
                        Op::Mul => lhs
                            .checked_mul(rhs)
                            .unwrap_or_else(|| panic!("overflow evaluating expr '{}'", self.raw)),
                        Op::Div => {
                            if rhs == 0 {
                                panic!("division by zero evaluating expr '{}'", self.raw);
                            }
                            lhs / rhs
                        }
                    };
                    stack.push(v);
                }
            }
        }

        if stack.len() != 1 {
            panic!("invalid expr '{}': stack depth {}", self.raw, stack.len());
        }
        stack[0]
    }

    #[inline]
    fn vars(&self) -> &[Arc<str>] {
        &self.vars
    }
}

fn resolve_usize_or_panic(bindings: &TensorBindings, name: &str, ctx: &str) -> usize {
    if let Some(v) = bindings.get_int_global(name) {
        return v;
    }
    if let Some(s) = bindings.get_var(name)
        && let Ok(v) = s.parse::<usize>()
    {
        return v;
    }
    panic!(
        "Missing required value '{name}' while evaluating expr '{ctx}'. Add it to the DSL (architecture.prepare.globals/derived_globals) or pass a runtime override."
    );
}

/// A parsed integer expression used by the DSL to define tensor dims/globals.
///
/// In JSON this can be:
/// - number: `128`
/// - string: `"m * d_model"` or `"{m} * {d_model}"`
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntExpr {
    Literal(usize),
    Parsed(Arc<ParsedIntExpr>),
}

impl IntExpr {
    #[inline]
    pub fn eval(&self, bindings: &TensorBindings) -> usize {
        match self {
            IntExpr::Literal(v) => *v,
            IntExpr::Parsed(expr) => expr.eval(bindings),
        }
    }

    #[inline]
    pub fn vars(&self) -> &[Arc<str>] {
        match self {
            IntExpr::Literal(_) => &[],
            IntExpr::Parsed(expr) => expr.vars(),
        }
    }
}

impl<'de> Deserialize<'de> for IntExpr {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct V;

        impl<'de> serde::de::Visitor<'de> for V {
            type Value = IntExpr;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a usize literal or an expression string")
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(IntExpr::Literal(value as usize))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                if value < 0 {
                    return Err(E::custom("negative integer not allowed in IntExpr"));
                }
                Ok(IntExpr::Literal(value as usize))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                let parsed = ParsedIntExpr::parse(value).map_err(E::custom)?;
                Ok(IntExpr::Parsed(Arc::new(parsed)))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                self.visit_str(&value)
            }
        }

        deserializer.deserialize_any(V)
    }
}

#[path = "int_expr.test.rs"]
mod tests;
