package cs362;

import java.util.ArrayList;
import java.util.List;

public abstract class KernelLogisticRegression extends Predictor {
	int totalInstances;
	int iteration;
	double learning_rate;
	double [][] gramMatrix;
	double [] alpha;
	double []cachealphaGramMatrix;
	List<Instance> training_instances;
	public KernelLogisticRegression() {
		int gradient_ascent_training_iterations = 5;
		if (CommandLineUtilities.hasArg("gradient_ascent_training_iterations"))
		gradient_ascent_training_iterations =
		CommandLineUtilities.getOptionValueAsInt("gradient_ascent_training_iterations");
		
		double gradient_ascent_learning_rate = 0.01;
		if (CommandLineUtilities.hasArg("gradient_ascent_learning_rate"))
		gradient_ascent_learning_rate =
		CommandLineUtilities.getOptionValueAsFloat("gradient_ascent_learning_rate");
		
		iteration = gradient_ascent_training_iterations;
		learning_rate = gradient_ascent_learning_rate;
	}
	public abstract double KernelMethod(FeatureVector x, FeatureVector xprime);
	public void cacheGramMatrix(List<Instance> instances, double[][] gramMatrix, int totalInstances){
		// compute and cache K(xi,xj)
		for(int i=0;i<totalInstances;i++)
			for(int j=0;j<totalInstances;j++)
				gramMatrix[i][j] = KernelMethod(instances.get(i).getFeatureVector(), instances.get(j).getFeatureVector());
	}
	// compute and cache sumation of alpha*K(xj,xi)
	public void cachealphaGramMatrix(double alpha[], double gramMatrix[][]){
		int len = gramMatrix.length;
		for(int i=0;i<len;i++){
			double sum = 0;
			for(int j=0;j<len;j++){
				sum += alpha[j]*gramMatrix[j][i];
			}
			cachealphaGramMatrix[i] = sum;
		}
	}
	public double linkFunction(double z){
		return (double)1.0/(1.0+Math.exp(-1*z)); 
	}
	@Override
    public void train ( List<Instance> instances )
    {
		// initialization
		training_instances = instances;
		totalInstances = instances.size();
		alpha = new double[totalInstances];
		gramMatrix = new double[totalInstances][totalInstances];
		cachealphaGramMatrix = new double[totalInstances];
		cacheGramMatrix(training_instances, gramMatrix, totalInstances);
		double gradient;
		for(;iteration>0;iteration--){
			cachealphaGramMatrix(alpha, gramMatrix);
			for(int i=0;i<totalInstances;i++){
				gradient = 0;
				//compute gradient
		        for(int j=0;j<totalInstances;j++)
		        {
		            double labelValue = Double.parseDouble(instances.get(j).getLabel().toString());
		            if(labelValue==1)
		            {
		                double lfVal = linkFunction(-1*cachealphaGramMatrix[j]);
		                gradient += lfVal * gramMatrix[j][i];
		            }
		            else if(labelValue==0)
		            {
		                double lfVal = linkFunction(cachealphaGramMatrix[j]);
		                gradient += lfVal*(-1*gramMatrix[j][i]);
		            }
		        }
		        //update alpha
		        alpha[i] += learning_rate*gradient;
			}
		}
    }
	@Override
    public Label predict ( Instance instance )
    {
		double sum = 0;
		double lfVal = 0;
		// compute alpha*K(xj,xi)
		for(int i=0;i<totalInstances;i++)
			sum = sum+alpha[i]*KernelMethod(training_instances.get(i).getFeatureVector(), instance.getFeatureVector());
		lfVal = linkFunction(sum);
		return lfVal>=0.5 ? new ClassificationLabel(1) : new ClassificationLabel(0);
    }
}
