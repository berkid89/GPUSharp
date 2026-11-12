using GPUSharp;

public static class Calculator
{
    [GpuKernel]
    public static float Add(float a, float b)
    {
        return a + b;
    }
}
