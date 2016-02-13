package cs362;

public class LinearKernelLogisticRegression extends KernelLogisticRegression{

	@Override
	public double KernelMethod(FeatureVector x, FeatureVector xprime) {
		// TODO Auto-generated method stub
		double res = 0;
        for(Integer f: x.featureVector.keySet())
            if( xprime.featureVector.keySet().contains( f ) )
                res += x.get(f)*xprime.get(f);
        return res;
	}

}
