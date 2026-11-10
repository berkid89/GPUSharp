use gpusharp_compiler::analyze_and_emit;

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

#[test]
fn add_emits_checked_rust_function() {
    let rust = analyze_and_emit(ADD).expect("Add should be supported");
    assert_eq!(rust, "pub fn add(a: f32, b: f32) -> f32 {\n    a + b\n}\n");
    assert!(
        !rust.contains("GpuKernel"),
        "attribute names stay in the IR and are not emitted"
    );
}

#[test]
fn multiply_emits_a_different_rust_function() {
    let rust = analyze_and_emit(MULTIPLY).expect("Multiply should be supported");
    assert_eq!(
        rust,
        "pub fn multiply(a: f32, b: f32) -> f32 {\n    a * b\n}\n"
    );
    assert_ne!(analyze_and_emit(ADD).unwrap(), rust);
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

    let error = analyze_and_emit(source).expect_err("method calls are unsupported");
    let upstream = csharp_to_rust::analyze_method(source).expect_err("upstream rejects calls");
    assert_eq!(error, upstream);
    assert!(
        error.to_string().contains("method calls"),
        "expected a readable transpiler error, got {error}"
    );
}
