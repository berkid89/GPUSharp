//! Compile a `[GpuKernel]` C# method into an element-wise CUDA kernel.
//!
//! GPUSharp does not parse C#. [`compile_kernel`] asks `csharp-to-rust` to
//! check the method, keeps that crate's scalar function, and wraps it in a
//! cuda-oxide `#[kernel]`. [`build_host_dll`] compiles that kernel into a
//! library the generated .NET API can call.

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
    /// Float kernels are required on this path.
    NotF32,
    /// Compiling the kernel library failed.
    Host(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transpile(error) => write!(f, "{error}"),
            Self::MissingGpuKernel => f.write_str("method is missing the [GpuKernel] attribute"),
            Self::NotF32 => f.write_str("GPUSharp kernels use float"),
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

/// Check one `[GpuKernel]` method and return its cuda-oxide module.
///
/// The string is a `#[cuda_module]` whose scalar function is the Rust from
/// `csharp-to-rust`. A `#[kernel]` runs that scalar once per element. The
/// scalar body is not rewritten.
///
/// # Errors
///
/// Returns [`Error::Transpile`] when `csharp-to-rust` rejects the source.
/// Returns [`Error::MissingGpuKernel`] when the method has no `GpuKernel`
/// attribute.
pub fn compile_kernel(source: &str) -> Result<String, Error> {
    let method = checked_gpu_kernel(source)?;
    let kernel_name = format!("{}_kernel", method.rust_name());
    Ok(emit_device_module(
        &method,
        method.rust_name(),
        &kernel_name,
    ))
}

fn checked_gpu_kernel(source: &str) -> Result<Method, Error> {
    let method = csharp_to_rust::analyze_method(source)?;
    if !method.attributes().iter().any(|name| name == "GpuKernel") {
        return Err(Error::MissingGpuKernel);
    }
    Ok(method)
}

pub(crate) fn emit_device_module(method: &Method, module_name: &str, kernel_name: &str) -> String {
    let scalar_name = method.rust_name();
    let parameters = method.parameters();
    let mut signature = Vec::with_capacity(parameters.len() + 1);
    for parameter in parameters {
        signature.push(format!(
            "{}: &[{}]",
            parameter.rust_name(),
            rust_type_name(parameter.primitive_type())
        ));
    }
    signature.push(format!(
        "mut out: DisjointSlice<{}>",
        rust_type_name(method.return_type())
    ));
    let signature = signature.join(", ");
    let call_args = parameters
        .iter()
        .map(|parameter| format!("{}[i]", parameter.rust_name()))
        .collect::<Vec<_>>()
        .join(", ");

    let mut body = String::new();
    body.push_str("        let idx = thread::index_1d();\n");
    body.push_str("        let i = idx.get();\n");
    if let Some(first) = parameters.first() {
        body.push_str(&format!(
            "        if i >= {}.len() {{\n            return;\n        }}\n",
            first.rust_name()
        ));
    }
    body.push_str("        if let Some(slot) = out.get_mut(idx) {\n");
    body.push_str(&format!(
        "            *slot = {scalar_name}({call_args});\n"
    ));
    body.push_str("        }\n");

    let scalar = indent_lines(&csharp_to_rust::emit_rust(method));
    format!(
        "\
#[cuda_module]
mod {module_name} {{
    use cuda_device::{{DisjointSlice, kernel, thread}};

{scalar}
    #[kernel]
    pub fn {kernel_name}({signature}) {{
{body}    }}
}}
"
    )
}

fn indent_lines(source: &str) -> String {
    let mut indented = String::new();
    for line in source.lines() {
        if !line.is_empty() {
            indented.push_str("    ");
        }
        indented.push_str(line);
        indented.push('\n');
    }
    indented
}

fn rust_type_name(primitive_type: PrimitiveType) -> &'static str {
    match primitive_type {
        PrimitiveType::I32 => "i32",
        PrimitiveType::F32 => "f32",
        PrimitiveType::F64 => "f64",
        PrimitiveType::Bool => "bool",
    }
}
