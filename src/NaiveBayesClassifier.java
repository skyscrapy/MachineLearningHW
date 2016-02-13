package cs362;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class NaiveBayesClassifier extends Predictor{
	double w[][];
	int totalFeatures;
	int totalInstances;
	double pc1;
	double pcx1;
	double lambda;
	public NaiveBayesClassifier (double lambda)
    {
        lambda  = lambda;
        pc1 = 0;
    }
	@Override
	public void train(List<Instance> instances) {
		// TODO Auto-generated method stub
		totalFeatures = getTotalFeatures(instances);
		totalInstances = instances.size();
		// p(c=1)
		int sumofpc1 = 0;
		// p(c=0)
		int sumofpc0 = 0;
		int sumofx1[] = new int [totalFeatures+1];
		int sumofx0[] = new int [totalFeatures+1];
		w = new double[totalFeatures+1][2];
		// get the number of xi|C=1&C=0
		for(Instance instance:instances){
			if(instance.getLabel().toString().equals("1")){
				sumofpc1++;
				for(Integer index:instance.getFeatureVector().featureVector.keySet()){
					if(instance.getFeatureVector().featureVector.get(index)>=0.5)
						sumofx1[index]++;
				}			
			}else{
				sumofpc0++;
				for(Integer index:instance.getFeatureVector().featureVector.keySet()){
					if(instance.getFeatureVector().featureVector.get(index)>=0.5){
						sumofx0[index]++;
					}
				}
			}
		}
		pc1 = (double)(sumofpc1+lambda)/(totalInstances+2*lambda);
		for(int i=1;i<totalFeatures+1;i++){
			w[i][0] = (double)(sumofx1[i]+lambda)/(sumofpc1+2*lambda);
			w[i][1] =  (double)(sumofx0[i]+lambda)/(sumofpc0+2*lambda);
			
		}
		/*for(int i=1;i<totalFeatures+1;i++){
			System.out.print(w[i][0]+" ");
		}
		System.out.println();
		for(int i=1;i<totalFeatures+1;i++){
			System.out.print(w[i][1]+" ");
		}*/
	}
	@Override
	public Label predict(Instance instance) {
		// TODO Auto-generated method stub
		double y1 = 0;
		double y2 = 0;
		for(Integer index:instance.getFeatureVector().featureVector.keySet()){
			if(index>totalFeatures)
				continue;
			else{
				if(instance.getFeatureVector().featureVector.get(index)>=0.5){
					y1+=Math.log(w[index][0]);
					y2+=Math.log(w[index][1]);
				}
				else{
					y1+=Math.log(1-w[index][0]);
					y2+=Math.log(1-w[index][1]);
				}
			}
		}
		y1+=Math.log(pc1);
		y2+=Math.log(1-pc1);
		if(y1>=y2)
			return new ClassificationLabel(1);
		else
			return new ClassificationLabel(0);
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
