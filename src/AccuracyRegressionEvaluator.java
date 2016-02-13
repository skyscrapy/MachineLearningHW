package cs362;

import java.util.List;

public class AccuracyRegressionEvaluator extends Evaluator{
	@Override
	public double evaluate(List<Instance> instances, Predictor predictor) {
		// TODO Auto-generated method stub
		double total_error = 0;
		for(Instance instance: instances){
			double predicted_label = Double.parseDouble(predictor.predict(instance).toString());
			double true_label = Double.parseDouble(instance.getLabel().toString());
			System.out.println(predicted_label+"  "+true_label);
			total_error += Math.abs(predicted_label-true_label);
		}
		return (double)total_error/(double)instances.size();
	}

}