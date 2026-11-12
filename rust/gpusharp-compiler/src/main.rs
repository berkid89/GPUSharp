use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        None => print_sample(),
        Some("dll") => {
            if let Err(error) = dll_command(args) {
                eprintln!("gpusharp-compiler: {error}");
                std::process::exit(1);
            }
        }
        Some(command) => {
            eprintln!("gpusharp-compiler: unknown command {command}");
            std::process::exit(2);
        }
    }
}

fn print_sample() {
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

fn dll_command(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let out = match (args.next().as_deref(), args.next()) {
        (Some("--out"), Some(path)) => path,
        _ => {
            return Err(
                "usage: gpusharp-compiler dll --out <dll> <export-name> <method.cs>".to_string(),
            );
        }
    };

    let mut owned = Vec::new();
    while let Some(export_name) = args.next() {
        let Some(path) = args.next() else {
            return Err(format!("missing C# file for export {export_name}"));
        };
        let source = std::fs::read_to_string(&path)
            .map_err(|error| format!("failed to read {path}: {error}"))?;
        owned.push((export_name, source));
    }
    if owned.is_empty() {
        return Err(
            "usage: gpusharp-compiler dll --out <dll> <export-name> <method.cs>".to_string(),
        );
    }

    let kernels = owned
        .iter()
        .map(|(export_name, source)| gpusharp_compiler::HostDllKernel {
            source,
            export_name,
        })
        .collect::<Vec<_>>();
    gpusharp_compiler::build_host_dll(&kernels, Path::new(&out))
        .map(|_| ())
        .map_err(|error| error.to_string())
}
