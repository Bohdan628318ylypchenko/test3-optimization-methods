using MathNet.Numerics.LinearAlgebra;

namespace Func2Minimizers
{
    public abstract class FSFunc2Minimizer : Func2Minimizer
    {
        protected override Vector<double> CalcNextPoint(Vector<double> currentPoint, Func<Vector<double>, double> f)
        {
            return currentPoint - CalcGrad(f, currentPoint) * CalcLambda(f, currentPoint);
        }

        protected abstract double CalcLambda(Func<Vector<double>, double> f, Vector<double> point);
    }
}
