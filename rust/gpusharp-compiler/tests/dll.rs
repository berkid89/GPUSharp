use std::path::Path;

use gpusharp_compiler::{Error, HostDllKernel, build_host_dll};

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
fn non_float_kernel_is_rejected_before_cargo_oxide() {
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

#[test]
fn kernel_library_requires_linux() {
    if cfg!(target_os = "linux") {
        return;
    }
    let error = build_host_dll(
        &[HostDllKernel {
            source: ADD,
            export_name: "gpusharp_calculator_add",
        }],
        Path::new("unused.so"),
    )
    .expect_err("cuda-oxide kernel builds require Linux");
    let Error::Host(message) = error else {
        panic!("expected a host error, got {error}");
    };
    assert!(
        message.contains("Linux"),
        "expected the Linux requirement, got {message}"
    );
}
