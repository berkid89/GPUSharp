use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use csharp_to_rust::PrimitiveType;
use csharp_to_rust::ir::Method;

use crate::Error;

/// One `[GpuKernel]` method to compile into a host cdylib.
pub struct HostDllKernel<'a> {
    /// C# source containing that single method.
    pub source: &'a str,
    /// C export name. The generated C# `DllImport` uses the same name.
    pub export_name: &'a str,
}

/// Compile `[GpuKernel]` methods into one host cdylib.
///
/// Each method must be an `f32` kernel. The cdylib exports `export_name` as a
/// C function that takes one pointer per parameter, an output pointer, and a
/// length.
///
/// # Errors
///
/// Returns [`Error::Transpile`], [`Error::MissingGpuKernel`], or [`Error::NotF32`]
/// when a method cannot be compiled. Returns [`Error::Host`] when `rustc` fails
/// or an export name is not a Rust identifier.
pub fn build_host_dll(kernels: &[HostDllKernel<'_>], out_path: &Path) -> Result<(), Error> {
    if kernels.is_empty() {
        return Err(Error::Host("no kernels to compile".to_string()));
    }

    let mut rust_source = String::new();
    for kernel in kernels {
        validate_export_name(kernel.export_name)?;
        let method = checked_f32_kernel(kernel.source)?;
        rust_source.push_str(&crate::compile_kernel(kernel.source)?);
        rust_source.push('\n');
        rust_source.push_str(&emit_c_abi(
            kernel.export_name,
            method.rust_name(),
            method.parameters().len(),
        ));
    }

    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|error| {
            Error::Host(format!("failed to create {}: {error}", parent.display()))
        })?;
    }

    let dir = TempDir(unique_temp_dir());
    fs::create_dir_all(&dir.0)
        .map_err(|error| Error::Host(format!("failed to create temp dir: {error}")))?;
    let source_path = dir.0.join("kernels.rs");
    fs::write(&source_path, rust_source)
        .map_err(|error| Error::Host(format!("failed to write kernel source: {error}")))?;
    compile_cdylib(&source_path, out_path)
}

fn checked_f32_kernel(source: &str) -> Result<Method, Error> {
    let method = csharp_to_rust::analyze_method(source)?;
    if !method.attributes().iter().any(|name| name == "GpuKernel") {
        return Err(Error::MissingGpuKernel);
    }
    if method.parameters().is_empty() {
        return Err(Error::Host(
            "element-wise host dll requires at least one float parameter".to_string(),
        ));
    }
    let f32_kernel = method.return_type() == PrimitiveType::F32
        && method
            .parameters()
            .iter()
            .all(|parameter| parameter.primitive_type() == PrimitiveType::F32);
    if !f32_kernel {
        return Err(Error::NotF32);
    }
    Ok(method)
}

fn validate_export_name(name: &str) -> Result<(), Error> {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return Err(Error::Host("export name is empty".to_string()));
    };
    if !(first == '_' || first.is_ascii_alphabetic())
        || !chars.all(|character| character == '_' || character.is_ascii_alphanumeric())
    {
        return Err(Error::Host(format!(
            "export name {name} is not a C identifier"
        )));
    }
    Ok(())
}

fn emit_c_abi(export_name: &str, rust_name: &str, parameter_count: usize) -> String {
    let mut signature = Vec::with_capacity(parameter_count + 2);
    let mut body = String::new();
    let mut arguments = Vec::with_capacity(parameter_count + 1);
    for index in 0..parameter_count {
        signature.push(format!("input_{index}: *const f32"));
        body.push_str(&format!(
            "    let input_{index} = unsafe {{ std::slice::from_raw_parts(input_{index}, len) }};\n"
        ));
        arguments.push(format!("input_{index}"));
    }
    signature.push("out: *mut f32".to_string());
    signature.push("len: usize".to_string());
    body.push_str("    let out = unsafe { std::slice::from_raw_parts_mut(out, len) };\n");
    arguments.push("out".to_string());

    format!(
        "#[unsafe(no_mangle)]\npub unsafe extern \"C\" fn {export_name}({}) {{\n{body}    {rust_name}_kernel({});\n}}\n",
        signature.join(", "),
        arguments.join(", ")
    )
}

fn compile_cdylib(source_path: &Path, out_path: &Path) -> Result<(), Error> {
    let output = Command::new("rustc")
        .arg("--edition")
        .arg("2024")
        .arg("--crate-type")
        .arg("cdylib")
        .arg("--crate-name")
        .arg("gpusharp_kernels")
        .arg(source_path)
        .arg("-o")
        .arg(out_path)
        .output()
        .map_err(|error| Error::Host(format!("failed to start rustc: {error}")))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(Error::Host(format!(
            "rustc failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        )))
    }
}

fn unique_temp_dir() -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("gpusharp-dll-{}-{nanos}-{id}", std::process::id()))
}

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::{HostDllKernel, build_host_dll};
    use crate::Error;
    use std::path::Path;

    #[test]
    fn non_float_kernel_is_rejected_before_rustc() {
        let source = r#"
            public static class Calculator
            {
                [GpuKernel]
                public static int Add(int a, int b)
                {
                    return a + b;
                }
            }
        "#;
        let error = build_host_dll(
            &[HostDllKernel {
                source,
                export_name: "gpusharp_calculator_add",
            }],
            Path::new("unused.dll"),
        )
        .expect_err("int kernels are not part of the host dll");
        assert_eq!(error, Error::NotF32);
    }
}
