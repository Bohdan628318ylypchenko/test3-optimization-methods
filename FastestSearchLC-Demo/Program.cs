using FastestSearch;
using MathNet.Numerics.LinearAlgebra;

class Program
{
    static void Main(string[] args)
    {
        FSFunc2Minimizer fs = new FSLcFunc2Minimizer();

        Func<Vector<double>, double> f =
            v => 3.0 * Math.Pow(v[0] - 14.0, 2) + v[0] * v[1] + 7.0 * Math.Pow(v[1], 2);
        Vector<double> startPoint = CreateVector.Dense(new double[] { 21.8, 21.8 });

        var result = fs.MinimizeFunc(f, startPoint, 1.725);
        foreach(var r in result)
        {
            Console.WriteLine(r.At(0).ToString() + " " + r.At(1).ToString());
        }      
    }
}