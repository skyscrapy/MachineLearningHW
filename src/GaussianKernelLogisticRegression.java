package cs362;

public class GaussianKernelLogisticRegression extends KernelLogisticRegression{
	double gaussian_kernel_sigma ;
	public GaussianKernelLogisticRegression(double sigma){
		this.gaussian_kernel_sigma = sigma;
	}
	@Override
	public double KernelMethod(FeatureVector x, FeatureVector xprime) {
		// TODO Auto-generated method stub
		double twoNorm = 0;
        for(Integer feature : x.featureVector.keySet())
            if(xprime.featureVector.keySet().contains(feature))
                twoNorm += Math.pow(x.get(feature)-xprime.get(feature),2);
            else
                twoNorm += Math.pow(x.get(feature),2);
        for(Integer feature : xprime.featureVector.keySet())
            if(!x.featureVector.keySet().contains(feature))
                twoNorm += Math.pow(xprime.get(feature),2);
       // twoNorm = Math.sqrt(twoNorm);
		return Math.exp(-1*twoNorm/(2*Math.pow(gaussian_kernel_sigma,2)));
	}
}
