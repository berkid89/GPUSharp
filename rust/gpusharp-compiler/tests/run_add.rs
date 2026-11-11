use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

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

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn emitted_add_kernel_adds_two_arrays() {
    let kernel = gpusharp_compiler::compile_kernel(ADD).expect("Add should be supported");
    let program = format!(
        "{kernel}\
         fn main() {{\n\
             let a = [1.0f32, 2.0, 3.0];\n\
             let b = [10.0f32, 20.0, 30.0];\n\
             let mut out = [0.0f32; 3];\n\
             add_kernel(&a, &b, &mut out);\n\
             println!(\"{{out:?}}\");\n\
         }}\n"
    );

    let dir = TempDir(unique_temp_dir());
    fs::create_dir_all(&dir.0).expect("temp directory should be created");
    let source_path = dir.0.join("add_kernel.rs");
    fs::write(&source_path, program).expect("source should be written");

    let binary_path = dir.0.join(binary_name());
    compile(&source_path, &binary_path);
    let stdout = run(&binary_path);
    assert_eq!(stdout.trim(), "[11.0, 22.0, 33.0]");
}

fn unique_temp_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after the unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("gpusharp-run-add-{}-{nanos}", std::process::id()))
}

fn binary_name() -> &'static str {
    if cfg!(windows) {
        "add_kernel.exe"
    } else {
        "add_kernel"
    }
}

fn compile(source_path: &Path, binary_path: &Path) {
    let output = Command::new("rustc")
        .arg("--edition")
        .arg("2024")
        .arg(source_path)
        .arg("-o")
        .arg(binary_path)
        .output()
        .expect("rustc should be installed");
    assert!(
        output.status.success(),
        "rustc failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn run(binary_path: &Path) -> String {
    let output = Command::new(binary_path)
        .output()
        .expect("compiled kernel should run");
    assert!(
        output.status.success(),
        "kernel failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("kernel stdout should be utf-8")
}
