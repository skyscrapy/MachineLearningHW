package cs362;

import java.util.Iterator;
import java.util.List;
import org.apache.commons.math3.linear.*;

import java.util.Iterator;
public class LinearRegression extends Predictor {
	RealVector W;
	RealMatrix MX;
	Label HX;
	int totalFeatures;
	@Override
	public void train(List<Instance> instances) {
		// TODO Auto-generated method stub
		double x[][];
		double y[];
		int totalInstances;
		totalFeatures = getTotalFeatures(instances);
		System.out.println(totalFeatures);
		totalInstances = instances.size();
		x = new double[totalInstances][totalFeatures+1];
		y = new double[totalInstances];
		for(int i=0;i<totalInstances;i++){
			y[i] = Double.parseDouble(instances.get(i).getLabel().toString());
			for(int j=0;j<totalFeatures+1;j++)
				if(j==0)
					x[i][j] = 1;
				else
					if(instances.get(i).getFeatureVector().featureVector.containsKey(j))
						x[i][j] = instances.get(i).getFeatureVector().featureVector.get(j);
					else
						x[i][j] = 0;			
		}
		MX = new Array2DRowRealMatrix(x);
		RealVector MY = new ArrayRealVector(y);		
		RealMatrix temp = MX.transpose().multiply(MX);
		DecompositionSolver solver = new LUDecomposition(temp).getSolver();
		temp = solver.getInverse();
		W = temp.multiply(MX.transpose()).operate(MY);
	}

	@Override
	public Label predict(Instance instance) {
		// TODO Auto-generated method stub
		double xArray[] = new double[totalFeatures+1];
		for(Integer index: instance.getFeatureVector().featureVector.keySet())
			xArray[index] = instance.getFeatureVector().get(index);
		double wArray[] = W.toArray();
		double val = wArray[0];
		for( int i=1;i<=totalFeatures;i++)
			val += wArray[i]*xArray[i];
		return new RegressionLabel(val);
	}
	public int getTotalFeatures(List<Instance> instances){
		int max = 0;
        for (Instance instance:instances)
            for (Integer index : instance.getFeatureVector().featureVector.keySet())
                if (index>max)
                	max = index;
        return max;
	}
}
