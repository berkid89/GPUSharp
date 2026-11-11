namespace GPUSharp;

/// Marks a scalar method as a GPU kernel. GPUSharp generates the matching array API.
[AttributeUsage(AttributeTargets.Method, Inherited = false, AllowMultiple = false)]
public sealed class GpuKernelAttribute : Attribute;
