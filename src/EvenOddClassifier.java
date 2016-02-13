package cs362;

import java.util.List;

public class EvenOddClassifier extends Predictor {
	ClassificationLabel evenoddLabel;
	@Override
	public void train(List<Instance> instances) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Label predict(Instance instance) {
		// TODO Auto-generated method stub
		double even_sum = 0;
		double odd_sum = 0;
		FeatureVector fv = instance.getFeatureVector();
		// calculate even_sum and odd_sum
		for(Integer index: fv.featureVector.keySet())
			if(index%2==0)
				even_sum+=fv.get(index);
			else
				odd_sum+=fv.get(index);
		if(even_sum>=odd_sum)
			evenoddLabel = new ClassificationLabel(1);
		else
			evenoddLabel = new ClassificationLabel(0);
		return evenoddLabel;
	}

}
