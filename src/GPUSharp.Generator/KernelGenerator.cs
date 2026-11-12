using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Text;

namespace GPUSharp.Generator;

/// <summary>
/// Builds a GPU kernel for each <c>[GpuKernel]</c> method and emits the array companion.
/// </summary>
[Generator]
public sealed class KernelGenerator : IIncrementalGenerator
{
    private const string MissingBuild = "GPUS0001";
    private const string UnsupportedKernel = "GPUS0002";
    private const string CompilerFailed = "GPUS0003";

    /// <inheritdoc />
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var kernels = context.SyntaxProvider
            .CreateSyntaxProvider(static (node, _) => node is MethodDeclarationSyntax, static (syntax, _) => Describe(syntax))
            .Where(static kernel => kernel is not null)
            .Select(static (kernel, _) => kernel!);

        var options = context.AnalyzerConfigOptionsProvider.Select(static (provider, _) => new BuildOptions(
            ReadOption(provider.GlobalOptions, "GpuSharpCompiler"),
            ReadOption(provider.GlobalOptions, "GpuSharpNativePath")));

        context.RegisterSourceOutput(kernels.Collect().Combine(options), static (output, input) =>
        {
            Generate(output, input.Left, input.Right);
        });
    }

    private static void Generate(SourceProductionContext output, ImmutableArray<KernelModel> kernels, BuildOptions options)
    {
        if (kernels.IsDefaultOrEmpty)
        {
            return;
        }

        var accepted = new List<KernelModel>();
        foreach (var kernel in kernels)
        {
            if (kernel.Diagnostic is null)
            {
                accepted.Add(kernel);
            }
            else
            {
                output.ReportDiagnostic(Diagnostic.Create(kernel.Diagnostic, kernel.Location, kernel.DiagnosticArguments));
            }
        }

        if (accepted.Count == 0)
        {
            return;
        }

        if (string.IsNullOrWhiteSpace(options.Compiler) || string.IsNullOrWhiteSpace(options.NativePath))
        {
            output.ReportDiagnostic(Diagnostic.Create(
                Descriptor(MissingBuild, "GPUSharp compiler path is missing. Import GPUSharp.targets."),
                accepted[0].Location));
            return;
        }

        string compiler = options.Compiler!;
        string nativePath = options.NativePath!;
        string nativeDirectory = Path.GetDirectoryName(nativePath) ?? ".";
        Directory.CreateDirectory(nativeDirectory);
        var arguments = new List<string> { "dll", "--out", nativePath };
        for (int index = 0; index < accepted.Count; index++)
        {
            KernelModel kernel = accepted[index];
            string csharpPath = Path.Combine(nativeDirectory, "kernel-" + index + ".cs");
            File.WriteAllText(csharpPath, kernel.CsharpSource);
            arguments.Add(kernel.ExportName);
            arguments.Add(csharpPath);
        }

        if (!RunCompiler(compiler, arguments, out string message) || !File.Exists(nativePath))
        {
            if (message.Length == 0)
            {
                message = "gpusharp-compiler did not write " + nativePath;
            }

            output.ReportDiagnostic(Diagnostic.Create(
                Descriptor(CompilerFailed, "GPUSharp kernel build failed: {0}"),
                accepted[0].Location,
                message));
            return;
        }

        output.AddSource("GPUSharp.Kernels.g.cs", SourceText.From(EmitCompanions(accepted), Encoding.UTF8));
    }

    private static bool RunCompiler(string compiler, List<string> arguments, out string message)
    {
        var start = new ProcessStartInfo
        {
            FileName = compiler,
            Arguments = string.Join(" ", arguments.Select(Quote)),
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };

        try
        {
            using var process = new Process { StartInfo = start };
            process.Start();
            var stdout = process.StandardOutput.ReadToEndAsync();
            var stderr = process.StandardError.ReadToEndAsync();
            process.WaitForExit();
            string output = (stderr.GetAwaiter().GetResult() + stdout.GetAwaiter().GetResult()).Trim();
            if (process.ExitCode == 0)
            {
                message = string.Empty;
                return true;
            }

            message = output.Length == 0 ? "gpusharp-compiler exited with code " + process.ExitCode : output;
            return false;
        }
        catch (Exception error)
        {
            message = error.Message;
            return false;
        }
    }

    private static string EmitCompanions(List<KernelModel> kernels)
    {
        var builder = new StringBuilder();
        builder.AppendLine("// <auto-generated/>");
        builder.AppendLine("#nullable enable");
        builder.AppendLine("using System;");
        builder.AppendLine("using System.Runtime.InteropServices;");

        foreach (var group in kernels.GroupBy(kernel => kernel.Namespace + "\n" + kernel.ClassName))
        {
            KernelModel first = group.First();
            if (first.Namespace.Length > 0)
            {
                builder.Append("namespace ").Append(first.Namespace).AppendLine();
                builder.AppendLine("{");
            }

            builder.Append("public static class ").Append(first.ClassName).AppendLine("Gpu");
            builder.AppendLine("{");
            foreach (KernelModel kernel in group)
            {
                AppendMethod(builder, kernel);
            }

            builder.AppendLine("}");
            if (first.Namespace.Length > 0)
            {
                builder.AppendLine("}");
            }
        }

        return builder.ToString();
    }

    private static void AppendMethod(StringBuilder builder, KernelModel kernel)
    {
        builder.Append("    public static float[] ").Append(kernel.MethodName).Append('(');
        for (int index = 0; index < kernel.ParameterNames.Length; index++)
        {
            if (index > 0)
            {
                builder.Append(", ");
            }

            builder.Append("float[] ").Append(kernel.ParameterNames[index]);
        }

        builder.AppendLine(")");
        builder.AppendLine("    {");
        foreach (string name in kernel.ParameterNames)
        {
            builder.Append("        if (").Append(name).AppendLine(" is null)");
            builder.AppendLine("        {");
            builder.Append("            throw new ArgumentNullException(nameof(").Append(name).AppendLine("));");
            builder.AppendLine("        }");
        }

        if (kernel.ParameterNames.Length > 1)
        {
            string first = kernel.ParameterNames[0];
            for (int index = 1; index < kernel.ParameterNames.Length; index++)
            {
                string name = kernel.ParameterNames[index];
                builder.Append("        if (").Append(first).Append(".Length != ").Append(name).AppendLine(".Length)");
                builder.AppendLine("        {");
                builder.AppendLine("            throw new ArgumentException(\"input arrays must have the same length\");");
                builder.AppendLine("        }");
            }
        }

        builder.Append("        float[] gpusharpOutput = new float[").Append(kernel.ParameterNames[0]).AppendLine(".Length];");
        for (int index = 0; index < kernel.ParameterNames.Length; index++)
        {
            builder.Append("        GCHandle gpusharpHandle").Append(index).Append(" = GCHandle.Alloc(")
                .Append(kernel.ParameterNames[index]).AppendLine(", GCHandleType.Pinned);");
        }

        builder.AppendLine("        GCHandle gpusharpOutputHandle = GCHandle.Alloc(gpusharpOutput, GCHandleType.Pinned);");
        builder.AppendLine("        try");
        builder.AppendLine("        {");
        builder.Append("            ").Append(kernel.MethodName).Append("Native(");
        for (int index = 0; index < kernel.ParameterNames.Length; index++)
        {
            builder.Append("gpusharpHandle").Append(index).Append(".AddrOfPinnedObject(), ");
        }

        builder.AppendLine("gpusharpOutputHandle.AddrOfPinnedObject(), (nuint)gpusharpOutput.Length);");
        builder.AppendLine("        }");
        builder.AppendLine("        finally");
        builder.AppendLine("        {");
        for (int index = 0; index < kernel.ParameterNames.Length; index++)
        {
            builder.Append("            gpusharpHandle").Append(index).AppendLine(".Free();");
        }

        builder.AppendLine("            gpusharpOutputHandle.Free();");
        builder.AppendLine("        }");
        builder.AppendLine("        return gpusharpOutput;");
        builder.AppendLine("    }");
        builder.Append("    [DllImport(\"gpusharp_kernels\", EntryPoint = \"").Append(kernel.ExportName)
            .AppendLine("\", CallingConvention = CallingConvention.Cdecl)]");
        builder.Append("    private static extern void ").Append(kernel.MethodName).Append("Native(");
        for (int index = 0; index < kernel.ParameterNames.Length; index++)
        {
            builder.Append("nint input").Append(index).Append(", ");
        }

        builder.AppendLine("nint output, nuint length);");
    }

    private static KernelModel? Describe(GeneratorSyntaxContext context)
    {
        var methodSyntax = (MethodDeclarationSyntax)context.Node;
        if (context.SemanticModel.GetDeclaredSymbol(methodSyntax) is not IMethodSymbol method)
        {
            return null;
        }

        if (!HasGpuKernel(method))
        {
            return null;
        }

        Location location = methodSyntax.Identifier.GetLocation();
        if (method.ContainingType is not INamedTypeSymbol type || type.TypeKind != TypeKind.Class)
        {
            return KernelModel.Invalid(location, Unsupported("GPUSharp kernels must be methods on a class."));
        }

        if (!method.IsStatic || method.DeclaredAccessibility != Accessibility.Public)
        {
            return KernelModel.Invalid(location, Unsupported("GPUSharp kernels must be public static methods."));
        }

        if (method.Parameters.Length == 0 || method.ReturnType.SpecialType != SpecialType.System_Single
            || method.Parameters.Any(parameter => parameter.Type.SpecialType != SpecialType.System_Single || parameter.RefKind != RefKind.None))
        {
            return KernelModel.Invalid(location, Unsupported("GPUSharp kernels must take and return float."));
        }

        string namespaceName = type.ContainingNamespace is { IsGlobalNamespace: false } containing
            ? containing.ToDisplayString()
            : string.Empty;
        string classKeyword = type.IsStatic ? "public static class " : "public class ";
        string declaration = methodSyntax.WithAttributeLists(default).ToString().Trim();
        string csharp = classKeyword + type.Name + "\n{\n    [GpuKernel]\n    " + declaration + "\n}\n";
        string[] names = method.Parameters.Select(parameter => parameter.Name).ToArray();
        return new KernelModel(
            namespaceName,
            type.Name,
            method.Name,
            "gpusharp_" + Snake(type.Name) + "_" + Snake(method.Name),
            csharp,
            names,
            location,
            null,
            Array.Empty<object>());
    }

    private static bool HasGpuKernel(IMethodSymbol method)
    {
        foreach (AttributeData attribute in method.GetAttributes())
        {
            INamedTypeSymbol? type = attribute.AttributeClass;
            if (type?.Name is "GpuKernel" or "GpuKernelAttribute"
                && type.ContainingNamespace?.ToDisplayString() == "GPUSharp")
            {
                return true;
            }
        }

        return false;
    }

    private static string? ReadOption(AnalyzerConfigOptions options, string name)
    {
        if (options.TryGetValue("build_property." + name, out string? value) && !string.IsNullOrWhiteSpace(value))
        {
            return value;
        }

        if (options.TryGetValue("build_property." + name.ToLowerInvariant(), out value) && !string.IsNullOrWhiteSpace(value))
        {
            return value;
        }

        return null;
    }

    private static DiagnosticDescriptor Unsupported(string message)
    {
        return Descriptor(UnsupportedKernel, message);
    }

    private static DiagnosticDescriptor Descriptor(string id, string messageFormat)
    {
        return new DiagnosticDescriptor(id, "GPUSharp", messageFormat, "GPUSharp", DiagnosticSeverity.Error, isEnabledByDefault: true);
    }

    private static string Snake(string name)
    {
        var builder = new StringBuilder();
        for (int index = 0; index < name.Length; index++)
        {
            char character = name[index];
            if (char.IsUpper(character) && index > 0)
            {
                builder.Append('_');
            }

            builder.Append(char.ToLowerInvariant(character));
        }

        return builder.ToString();
    }

    private static string Quote(string value)
    {
        return "\"" + value.Replace("\"", "\\\"") + "\"";
    }

    private sealed class BuildOptions
    {
        public BuildOptions(string? compiler, string? nativePath)
        {
            Compiler = compiler;
            NativePath = nativePath;
        }

        public string? Compiler { get; }

        public string? NativePath { get; }
    }

    private sealed class KernelModel
    {
        public KernelModel(
            string namespaceName,
            string className,
            string methodName,
            string exportName,
            string csharpSource,
            string[] parameterNames,
            Location location,
            DiagnosticDescriptor? diagnostic,
            object[] diagnosticArguments)
        {
            Namespace = namespaceName;
            ClassName = className;
            MethodName = methodName;
            ExportName = exportName;
            CsharpSource = csharpSource;
            ParameterNames = parameterNames;
            Location = location;
            Diagnostic = diagnostic;
            DiagnosticArguments = diagnosticArguments;
        }

        public string Namespace { get; }

        public string ClassName { get; }

        public string MethodName { get; }

        public string ExportName { get; }

        public string CsharpSource { get; }

        public string[] ParameterNames { get; }

        public Location Location { get; }

        public DiagnosticDescriptor? Diagnostic { get; }

        public object[] DiagnosticArguments { get; }

        public static KernelModel Invalid(Location location, DiagnosticDescriptor diagnostic)
        {
            return new KernelModel(string.Empty, string.Empty, string.Empty, string.Empty, string.Empty, Array.Empty<string>(), location, diagnostic, Array.Empty<object>());
        }
    }
}
