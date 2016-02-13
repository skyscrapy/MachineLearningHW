package cs362;

import java.util.HashMap;
import java.util.List;
import java.util.Random;

public class MajorityClassifier extends Predictor{
	HashMap<ClassificationLabel, Integer> map = new HashMap<>();
	ClassificationLabel majorityLabel;
	private ClassificationLabel [] mLabels;
	
	@Override
	public void train(List<Instance> instances) {
		// traverse labels in instances and record the appearing times into map
		for(Instance instance:instances){
			ClassificationLabel label = (ClassificationLabel) instance.getLabel();
			// if existing, add 1, else set 1
			if(map.get(label)!=null)
				map.put(label, map.get(label)+1);
			else
				map.put(label, 1);
		}
		mLabels = new ClassificationLabel[map.size()];
		int max = 0;
		// get max value
		for(ClassificationLabel traverseKey:map.keySet())
			if(map.get(traverseKey)>max)
				max = map.get(traverseKey);
		// index of array as well as the size of the array
		int i = 0;
		// store all labels with the same appearing times into an array
		for(ClassificationLabel traverseKey:map.keySet()){
			if(map.get(traverseKey)==max){
				mLabels[i] = traverseKey;
				i++;
			}
		}
		// randomize a number ranging from 0~(i-1) and assign mLables[x] to majorityLabel
		Random ran = new Random(); 
		int x = ran.nextInt(i);
		majorityLabel = mLabels[x];
	}

	@Override
	public Label predict(Instance instance) {
		return majorityLabel;
	}

}
