package cs362;

import java.util.List;

public class AccuracyEvaluator extends Evaluator{
	@Override
	public double evaluate(List<Instance> instances, Predictor predictor) {
		// TODO Auto-generated method stub
		double correct = 0.0;
		double total = 0.0;
		for (Instance instance : instances) {
			total += 1.0;
	//		System.out.println(instance.getLabel().toString());
			System.out.println(predictor.predict(instance).toString());
			if (!(instance.getLabel() == null) && predictor.predict(instance).toString().equals(instance.getLabel().toString())) {
				correct += 1.0;
			}
		}
		if (total == 0.0)
			return 0;
		System.out.println(correct+" "+total);
		return correct/total;
	}

}
