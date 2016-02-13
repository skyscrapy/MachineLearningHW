package cs362;

import java.util.List;

public class PerceptronClassifier extends Predictor{
	double learning_rate;
	int training_iterations;
	int totalFeatures;
	double w[];
	public PerceptronClassifier(double online_learning_rate, int online_training_iterations){
		learning_rate = online_learning_rate;
		training_iterations = online_training_iterations;
	}
	@Override
	public void train(List<Instance> instances) {
		totalFeatures = getTotalFeatures(instances);
		w = new double[totalFeatures+1];
		double x[];
		double predicted_y;
		double y;
		double val;
		// TODO Auto-generated method stub
		for(int k=1;k<=training_iterations;k++){	
			for(int i=0;i<instances.size();i++){
				x = new double[totalFeatures+1];
				val = 0;
				y = Integer.parseInt(instances.get(i).getLabel().toString());
				for(Integer index:instances.get(i).getFeatureVector().featureVector.keySet())
					x[index] = instances.get(i).getFeatureVector().get(index);
				for(int j=1;j<totalFeatures+1;j++)
					val += w[j]*x[j];
				if(val>=0)
					predicted_y = 1;
				else
					predicted_y = 0;
				if(predicted_y != y){
					if(y==0)
						y=-1;
					
					for(Integer index:instances.get(i).getFeatureVector().featureVector.keySet()){		
						w[index] = w[index]+(double)learning_rate*y*x[index];
					}
					
				
				}
			}
		}
	}

	@Override
	public Label predict(Instance instance) {
		// TODO Auto-generated method stub
		double xArray[] = new double[totalFeatures+1];
		double y = 0;
		for(Integer index: instance.getFeatureVector().featureVector.keySet())
			xArray[index] = instance.getFeatureVector().get(index);
		for(int i=1;i<totalFeatures+1;i++)
			y += xArray[i]*w[i]; 
		if(y>=0)
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
