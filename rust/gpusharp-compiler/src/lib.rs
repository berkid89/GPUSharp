//! Turn a `[GpuKernel]` C# method into an element-wise host Rust kernel.
//!
//! GPUSharp does not parse C#. [`compile_kernel`] asks `csharp-to-rust` to
//! check the method, keeps that crate's scalar function, and appends a slice
//! wrapper. [`build_host_dll`] compiles that wrapper into a CPU cdylib.
//! This does not generate PTX or run on a GPU.

mod dll;

pub use dll::{HostDllKernel, build_host_dll};

use csharp_to_rust::PrimitiveType;
use csharp_to_rust::ir::Method;

/// Why [`compile_kernel`] or [`build_host_dll`] failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The C# source is invalid or outside the `csharp-to-rust` subset.
    Transpile(csharp_to_rust::Error),
    /// The method type-checks but is not marked `[GpuKernel]`.
    MissingGpuKernel,
    /// The host dll path currently accepts only `float` kernels.
    NotF32,
    /// Compiling the host dll failed.
    Host(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transpile(error) => write!(f, "{error}"),
            Self::MissingGpuKernel => f.write_str("method is missing the [GpuKernel] attribute"),
            Self::NotF32 => {
                f.write_str("element-wise host dll currently supports float kernels only")
            }
            Self::Host(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transpile(error) => Some(error),
            Self::MissingGpuKernel | Self::NotF32 | Self::Host(_) => None,
        }
    }
}

impl From<csharp_to_rust::Error> for Error {
    fn from(error: csharp_to_rust::Error) -> Self {
        Self::Transpile(error)
    }
}

/// Check one `[GpuKernel]` method and return host Rust for it.
///
/// The string is the scalar function from `csharp-to-rust`, then a blank line,
/// then an element-wise function over slices. The scalar body is not rewritten.
///
/// # Errors
///
/// Returns [`Error::Transpile`] when `csharp-to-rust` rejects the source.
/// Returns [`Error::MissingGpuKernel`] when the method has no `GpuKernel`
/// attribute.
pub fn compile_kernel(source: &str) -> Result<String, Error> {
    let method = csharp_to_rust::analyze_method(source)?;
    if !method.attributes().iter().any(|name| name == "GpuKernel") {
        return Err(Error::MissingGpuKernel);
    }
    let scalar = csharp_to_rust::emit_rust(&method);
    let kernel = emit_elementwise_kernel(&method);
    Ok(format!("{scalar}\n{kernel}"))
}

fn emit_elementwise_kernel(method: &Method) -> String {
    let name = method.rust_name();
    let return_type = rust_type_name(method.return_type());
    let parameters = method.parameters();

    let mut inputs = Vec::with_capacity(parameters.len() + 1);
    for parameter in parameters {
        inputs.push(format!(
            "{}: &[{}]",
            parameter.rust_name(),
            rust_type_name(parameter.primitive_type())
        ));
    }
    inputs.push(format!("out: &mut [{return_type}]"));

    let mut body = String::new();
    for parameter in parameters {
        body.push_str(&format!(
            "    assert_eq!({}.len(), out.len());\n",
            parameter.rust_name()
        ));
    }

    let call_args = parameters
        .iter()
        .map(|parameter| format!("{}[i]", parameter.rust_name()))
        .collect::<Vec<_>>()
        .join(", ");
    body.push_str("    for i in 0..out.len() {\n");
    body.push_str(&format!("        out[i] = {name}({call_args});\n"));
    body.push_str("    }\n");

    format!("pub fn {name}_kernel({}) {{\n{body}}}\n", inputs.join(", "))
}

fn rust_type_name(primitive_type: PrimitiveType) -> &'static str {
    match primitive_type {
        PrimitiveType::I32 => "i32",
        PrimitiveType::F32 => "f32",
        PrimitiveType::F64 => "f64",
        PrimitiveType::Bool => "bool",
    }
}
