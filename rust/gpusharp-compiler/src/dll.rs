use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use csharp_to_rust::PrimitiveType;
use csharp_to_rust::ir::Method;

use crate::Error;

const OXIDE_REPOSITORY: &str = "https://github.com/NVlabs/cuda-oxide.git";
const OXIDE_TAG: &str = "v0.2.1";

/// One `[GpuKernel]` method to compile into a kernel library.
pub struct HostDllKernel<'a> {
    /// C# source containing that single method.
    pub source: &'a str,
    /// C export name. The generated C# `DllImport` uses the same name.
    pub export_name: &'a str,
}

/// Compile `[GpuKernel]` methods into one CUDA kernel library.
///
/// Each method must be an `f32` kernel. The cdylib exports `export_name` as a
/// C function that takes one pointer per parameter, an output pointer, and a
/// length. That function copies the arrays to the GPU, launches the kernel,
/// and copies the result back. The device module is read from this shared
/// library. `cuda-oxide` otherwise looks in the host process executable, which
/// is `dotnet` when .NET calls the export.
///
/// The library is built with `cargo oxide` on Linux. `GPUSHARP_CUDA_ARCH`
/// selects the PTX architecture and defaults to `sm_80`. `GPUSHARP_OXIDE_TARGET`
/// selects the Cargo target directory used to reuse cuda-oxide builds.
///
/// # Errors
///
/// Returns [`Error::Transpile`], [`Error::MissingGpuKernel`], or [`Error::NotF32`]
/// when a method cannot be compiled. Returns [`Error::Host`] when `cargo oxide`
/// fails or an export name is not a C identifier. On a non-Linux host the
/// kernels are still checked, and the result is [`HostDllBuild::Deferred`]:
/// the managed API can build, and calling it throws.
pub fn build_host_dll(
    kernels: &[HostDllKernel<'_>],
    out_path: &Path,
) -> Result<HostDllBuild, Error> {
    let rust_source = library_source(kernels)?;
    if !cfg!(target_os = "linux") {
        return Ok(HostDllBuild::Deferred);
    }

    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|error| {
            Error::Host(format!("failed to create {}: {error}", parent.display()))
        })?;
    }

    let dir = TempDir(unique_temp_dir());
    write_oxide_project(&dir.0, &rust_source)?;
    compile_with_cargo_oxide(&dir.0, out_path)?;
    Ok(HostDllBuild::Compiled)
}

/// Whether [`build_host_dll`] wrote a kernel library.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostDllBuild {
    /// `out_path` is a CUDA kernel library.
    Compiled,
    /// The kernels are valid, but this host cannot compile them.
    Deferred,
}

pub(crate) fn library_source(kernels: &[HostDllKernel<'_>]) -> Result<String, Error> {
    if kernels.is_empty() {
        return Err(Error::Host("no kernels to compile".to_string()));
    }

    let mut rust_source = String::from("use cuda_host::cuda_module;\n\n");
    rust_source.push_str(host_loader());
    rust_source.push('\n');
    for kernel in kernels {
        validate_export_name(kernel.export_name)?;
        let method = checked_f32_kernel(kernel.source)?;
        rust_source.push_str(&crate::emit_device_module(
            &method,
            &format!("{}_module", kernel.export_name),
            &format!("{}_kernel", kernel.export_name),
        ));
        rust_source.push('\n');
        rust_source.push_str(&emit_cuda_export(
            kernel.export_name,
            method.parameters().len(),
        ));
        rust_source.push('\n');
    }
    Ok(rust_source)
}

fn checked_f32_kernel(source: &str) -> Result<Method, Error> {
    let method = csharp_to_rust::analyze_method(source)?;
    if !method.attributes().iter().any(|name| name == "GpuKernel") {
        return Err(Error::MissingGpuKernel);
    }
    if method.parameters().is_empty() {
        return Err(Error::Host(
            "a GPUSharp kernel needs at least one float parameter".to_string(),
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

fn emit_cuda_export(export_name: &str, parameter_count: usize) -> String {
    let mut signature = Vec::with_capacity(parameter_count + 2);
    let mut slices = String::new();
    let mut copies = String::new();
    let mut launch_args = Vec::with_capacity(parameter_count + 1);
    for index in 0..parameter_count {
        signature.push(format!("input_{index}: *const f32"));
        slices.push_str(&format!(
            "    let input_{index} = unsafe {{ std::slice::from_raw_parts(input_{index}, len) }};\n"
        ));
        copies.push_str(&format!(
            "    let input_{index}_dev = cuda_core::DeviceBuffer::from_host(&stream, input_{index}).expect(\"GPUSharp could not copy an input to the GPU\");\n"
        ));
        launch_args.push(format!("&input_{index}_dev"));
    }
    signature.push("out: *mut f32".to_string());
    signature.push("len: usize".to_string());
    slices.push_str("    let out = unsafe { std::slice::from_raw_parts_mut(out, len) };\n");
    launch_args.push("&mut out_dev".to_string());

    format!(
        "\
#[unsafe(no_mangle)]
pub unsafe extern \"C\" fn {export_name}({}) {{
    if len == 0 {{
        return;
    }}
    let Ok(len_u32) = u32::try_from(len) else {{
        panic!(\"GPUSharp kernel length does not fit in u32\");
    }};
{slices}    let ctx = cuda_core::CudaContext::new(0).expect(\"GPUSharp could not open CUDA device 0\");
    let stream = ctx.default_stream();
{copies}    let mut out_dev = cuda_core::DeviceBuffer::<f32>::zeroed(&stream, len).expect(\"GPUSharp could not allocate the GPU output\");
    let cuda_module = gpusharp_load_cuda_module(&ctx);
    let module = {export_name}_module::from_module(cuda_module).expect(\"GPUSharp could not load the CUDA module\");
    module
        .{export_name}_kernel(
            &stream,
            cuda_core::LaunchConfig::for_num_elems(len_u32),
            {},
        )
        .expect(\"GPUSharp kernel launch failed\");
    out_dev
        .copy_to_host(&stream, out)
        .expect(\"GPUSharp could not copy the GPU output back\");
}}
",
        signature.join(", "),
        launch_args.join(", ")
    )
}

fn host_loader() -> &'static str {
    "\
#[repr(C)]
struct GpusharpDlInfo {
    dli_fname: *const std::ffi::c_char,
    dli_fbase: *mut std::ffi::c_void,
    dli_sname: *const std::ffi::c_char,
    dli_saddr: *mut std::ffi::c_void,
}

#[link(name = \"dl\")]
unsafe extern \"C\" {
    fn dladdr(addr: *const std::ffi::c_void, info: *mut GpusharpDlInfo) -> i32;
}

#[inline(never)]
fn gpusharp_shared_library_path() -> String {
    unsafe {
        let mut info = std::mem::zeroed::<GpusharpDlInfo>();
        let found = dladdr(
            gpusharp_shared_library_path as *const std::ffi::c_void,
            &mut info,
        );
        if found == 0 || info.dli_fname.is_null() {
            panic!(\"GPUSharp could not find its CUDA library\");
        }
        std::ffi::CStr::from_ptr(info.dli_fname)
            .to_string_lossy()
            .into_owned()
    }
}

fn gpusharp_load_cuda_module(
    ctx: &std::sync::Arc<cuda_core::CudaContext>,
) -> std::sync::Arc<cuda_core::CudaModule> {
    let path = gpusharp_shared_library_path();
    let bundles = cuda_core::embedded::artifact_bundles_from_binary_path(&path).unwrap_or_else(
        |error| panic!(\"GPUSharp could not read the CUDA module from {path}: {error}\"),
    );
    let name = env!(\"CARGO_PKG_NAME\");
    let bundle = bundles
        .into_iter()
        .find(|bundle| bundle.name == name)
        .unwrap_or_else(|| panic!(\"GPUSharp could not load the CUDA module {name} from {path}\"));
    let embedded = cuda_core::embedded::EmbeddedModule::new(bundle).unwrap_or_else(|| {
        panic!(\"GPUSharp could not load the CUDA module {name} from {path}\")
    });
    embedded.load(ctx).unwrap_or_else(|error| {
        panic!(\"GPUSharp could not load the CUDA module {name} from {path}: {error}\")
    })
}
"
}

fn write_oxide_project(dir: &Path, rust_source: &str) -> Result<(), Error> {
    fs::create_dir_all(dir.join("src"))
        .map_err(|error| Error::Host(format!("failed to create temp dir: {error}")))?;
    fs::write(dir.join("Cargo.toml"), cargo_toml())
        .map_err(|error| Error::Host(format!("failed to write Cargo.toml: {error}")))?;
    fs::write(dir.join("rust-toolchain.toml"), rust_toolchain_toml())
        .map_err(|error| Error::Host(format!("failed to write rust-toolchain.toml: {error}")))?;
    fs::write(dir.join("src").join("lib.rs"), rust_source)
        .map_err(|error| Error::Host(format!("failed to write kernel source: {error}")))
}

fn cargo_toml() -> String {
    format!(
        "\
[package]
name = \"gpusharp_kernels\"
version = \"0.0.0\"
edition = \"2024\"
publish = false

[workspace]

[lib]
crate-type = [\"cdylib\"]

[dependencies]
cuda-device = {{ git = \"{OXIDE_REPOSITORY}\", tag = \"{OXIDE_TAG}\" }}
cuda-host = {{ git = \"{OXIDE_REPOSITORY}\", tag = \"{OXIDE_TAG}\" }}
cuda-core = {{ git = \"{OXIDE_REPOSITORY}\", tag = \"{OXIDE_TAG}\" }}
"
    )
}

fn rust_toolchain_toml() -> &'static str {
    "\
[toolchain]
channel = \"nightly-2026-04-03\"
components = [\"rust-src\", \"rustc-dev\", \"rust-analyzer\", \"clippy\", \"llvm-tools\"]
"
}

fn compile_with_cargo_oxide(project_dir: &Path, out_path: &Path) -> Result<(), Error> {
    let target_dir = oxide_target_dir()?;
    fs::create_dir_all(&target_dir).map_err(|error| {
        Error::Host(format!(
            "failed to create {}: {error}",
            target_dir.display()
        ))
    })?;
    let arch = cuda_arch();
    let output = Command::new("cargo")
        .arg("oxide")
        .arg("build")
        .arg("--arch")
        .arg(&arch)
        .current_dir(project_dir)
        .env("CUDA_OXIDE_TARGET", &arch)
        .env("CARGO_TARGET_DIR", &target_dir)
        .env("RUSTC_BOOTSTRAP", "1")
        .output()
        .map_err(|error| Error::Host(format!("failed to start cargo oxide: {error}")))?;
    if !output.status.success() {
        return Err(Error::Host(cargo_oxide_failure(&output)));
    }

    let built = target_dir.join("release").join("libgpusharp_kernels.so");
    if !built.is_file() {
        return Err(Error::Host(format!(
            "cargo oxide did not write {}",
            built.display()
        )));
    }
    fs::copy(&built, out_path).map_err(|error| {
        Error::Host(format!(
            "failed to copy {} to {}: {error}",
            built.display(),
            out_path.display()
        ))
    })?;
    Ok(())
}

fn cargo_oxide_failure(output: &std::process::Output) -> String {
    let mut text = String::from_utf8_lossy(&output.stderr).into_owned();
    text.push(' ');
    text.push_str(&String::from_utf8_lossy(&output.stdout));
    let mut collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    const MAX_LEN: usize = 2_000;
    if collapsed.len() > MAX_LEN {
        let mut end = MAX_LEN;
        while !collapsed.is_char_boundary(end) {
            end -= 1;
        }
        collapsed.truncate(end);
        collapsed.push_str("...");
    }
    if collapsed.is_empty() {
        format!("cargo oxide failed with {}", output.status)
    } else {
        format!("cargo oxide failed: {collapsed}")
    }
}

fn cuda_arch() -> String {
    match std::env::var("GPUSHARP_CUDA_ARCH") {
        Ok(arch) if !arch.is_empty() => arch,
        _ => "sm_80".to_string(),
    }
}

fn oxide_target_dir() -> Result<PathBuf, Error> {
    if let Some(path) = std::env::var_os("GPUSHARP_OXIDE_TARGET")
        && !path.is_empty()
    {
        return Ok(PathBuf::from(path));
    }
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"));
    let Some(home) = home else {
        return Err(Error::Host(
            "GPUSharp could not find a home directory for the cuda-oxide build cache. Set GPUSHARP_OXIDE_TARGET.".to_string(),
        ));
    };
    Ok(PathBuf::from(home)
        .join(".cache")
        .join("gpusharp")
        .join("oxide-target"))
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
    use super::{HostDllKernel, library_source};

    const ADD: &str = r#"
        public static class Calculator
        {
            [GpuKernel]
            public static float Add(float a, float b)
            {
                return a + b;
            }
        }
    "#;

    #[test]
    fn cuda_library_source_launches_one_thread_per_element() {
        let source = library_source(&[HostDllKernel {
            source: ADD,
            export_name: "gpusharp_calculator_add",
        }])
        .expect("Add should be supported");
        assert!(source.contains("use cuda_host::cuda_module;"));
        assert!(source.contains("artifact_bundles_from_binary_path"));
        assert!(source.contains("dladdr"));
        assert!(!source.contains("_module::load("));
        assert!(source.contains("mod gpusharp_calculator_add_module"));
        assert!(source.contains("pub fn gpusharp_calculator_add_kernel"));
        assert!(source.contains("CudaContext::new(0)"));
        assert!(source.contains("LaunchConfig::for_num_elems"));
        assert!(source.contains("if len == 0"));
        assert!(source.contains("GPUSharp kernel length does not fit in u32"));
        assert!(source.contains("copy_to_host"));
    }
}
