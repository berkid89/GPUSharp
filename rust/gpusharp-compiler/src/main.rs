fn main() {
    let source = r#"
        public static class Calculator
        {
            [GpuKernel]
            public static float Add(float a, float b)
            {
                return a + b;
            }
        }
    "#;

    match gpusharp_compiler::compile_kernel(source) {
        Ok(rust) => print!("{rust}"),
        Err(error) => {
            eprintln!("gpusharp-compiler: {error}");
            std::process::exit(1);
        }
    }
}
