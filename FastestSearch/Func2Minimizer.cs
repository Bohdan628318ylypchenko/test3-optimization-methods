using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace Func2Minimizers
{
    public abstract class Func2Minimizer
    {
        public Vector<double>[] MinimizeFunc(Func<Vector<double>, double> f,
                                             Vector<double> startPoint,
                                             double e)
        {
            // List to store path
            LinkedList<Vector<double>> result = new LinkedList<Vector<double>>();
            result.AddLast(startPoint);

            // Minimizing
            Vector<double> currentPoint = startPoint;
            while(!StopCriteria(f, currentPoint, e))
            {
                currentPoint = CalcNextPoint(currentPoint, f);
                result.AddLast(currentPoint);
            }
            
            // Returning
            return result.ToArray();
        }      

        private bool StopCriteria(Func<Vector<double>, double> f,
                                  Vector<double> point,
                                  double e)
        {
            // Gradient module less than e
            return CalcGrad(f, point).L2Norm() <= e;
        }

        protected abstract Vector<double> CalcNextPoint(Vector<double> currentPoint, Func<Vector<double>, double> f);

        protected Vector<double> CalcGrad(Func<Vector<double>, double> f,
                                          Vector<double> point)
        {
            // Map copy 
            return point.MapIndexed<double>((i, x) => Differentiate.FirstPartialDerivative(p => f(CreateVector.Dense(p)), point.AsArray(), i));
        }

    }
}