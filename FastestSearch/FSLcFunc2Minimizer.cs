using MathNet.Numerics.LinearAlgebra;

namespace Func2Minimizers
{
    public class FSLcFunc2Minimizer : FSFunc2Minimizer
    {
        protected override double CalcLambda(Func<Vector<double>, double> f, Vector<double> point)
        {
            return 1.0 / CalcGrad(f, point).L2Norm();
        }
    }
}
