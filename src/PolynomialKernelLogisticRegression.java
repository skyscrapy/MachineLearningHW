package cs362;

public class PolynomialKernelLogisticRegression extends KernelLogisticRegression{
	double polynomial_kernel_exponent;
	public PolynomialKernelLogisticRegression(double exponent){
		polynomial_kernel_exponent = exponent;
	}
	@Override
	public double KernelMethod(FeatureVector x, FeatureVector xprime) {
		// TODO Auto-generated method stub
		double res = 0;
        for(Integer f:x.featureVector.keySet())
            if (xprime.featureVector.keySet().contains( f ) )
                res += x.get(f)*xprime.get(f);
		return Math.pow(1+res, polynomial_kernel_exponent);
	}

}
