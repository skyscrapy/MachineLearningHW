package cs362;

import java.io.Serializable;
import java.util.HashMap;

public class FeatureVector implements Serializable {
	public HashMap<Integer, Double>featureVector = new HashMap<>();
	public void add(int index, double value) {
		// TODO Auto-generated method stub
		featureVector.put(index, value);
	}
	
	public double get(int index) {
		// TODO Auto-generated method stub
		return featureVector.get(index);
	}
}
