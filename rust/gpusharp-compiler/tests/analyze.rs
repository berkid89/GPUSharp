use gpusharp_compiler::{Error, compile_kernel};

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

const MULTIPLY: &str = r#"
    public static class Calculator
    {
        [GpuKernel]
        public static float Multiply(float a, float b)
        {
            return a * b;
        }
    }
"#;

const ADD_RUST: &str = r#"pub fn add(a: f32, b: f32) -> f32 {
    a + b
}

pub fn add_kernel(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    assert_eq!(b.len(), out.len());
    for i in 0..out.len() {
        out[i] = add(a[i], b[i]);
    }
}
"#;

const MULTIPLY_RUST: &str = r#"pub fn multiply(a: f32, b: f32) -> f32 {
    a * b
}

pub fn multiply_kernel(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    assert_eq!(b.len(), out.len());
    for i in 0..out.len() {
        out[i] = multiply(a[i], b[i]);
    }
}
"#;

#[test]
fn add_emits_elementwise_host_kernel() {
    let rust = compile_kernel(ADD).expect("Add should be supported");
    assert_eq!(rust, ADD_RUST);
    assert!(
        !rust.contains("GpuKernel"),
        "attribute names stay in the IR and are not emitted"
    );
}

#[test]
fn multiply_emits_a_different_kernel() {
    let rust = compile_kernel(MULTIPLY).expect("Multiply should be supported");
    assert_eq!(rust, MULTIPLY_RUST);
    assert_ne!(compile_kernel(ADD).unwrap(), rust);
}

#[test]
fn missing_gpu_kernel_attribute_is_rejected() {
    let source = r#"
        public static class Calculator
        {
            public static float Add(float a, float b)
            {
                return a + b;
            }
        }
    "#;

    let error = compile_kernel(source).expect_err("GpuKernel is required");
    assert_eq!(error, Error::MissingGpuKernel);
    assert!(
        error.to_string().contains("[GpuKernel]"),
        "expected a readable attribute error, got {error}"
    );
}

#[test]
fn unsupported_csharp_returns_the_transpiler_error() {
    let source = r#"
        public static class Calculator
        {
            public static float Call(float value)
            {
                return Abs(value);
            }
        }
    "#;

    let error = compile_kernel(source).expect_err("method calls are unsupported");
    let Error::Transpile(error) = error else {
        panic!("expected the upstream error, got {error}");
    };
    let upstream = csharp_to_rust::analyze_method(source).expect_err("upstream rejects calls");
    assert_eq!(error, upstream);
    assert!(
        error.to_string().contains("method calls"),
        "expected a readable transpiler error, got {error}"
    );
}
