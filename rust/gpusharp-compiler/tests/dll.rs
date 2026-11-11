use std::path::Path;

use gpusharp_compiler::{Error, HostDllKernel, build_host_dll};

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
