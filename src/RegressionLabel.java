package cs362;

import java.io.Serializable;

public class RegressionLabel extends Label implements Serializable {
	public double rl;
	public RegressionLabel(double label) {
		// TODO Auto-generated constructor stub
		rl = label;
	}

	@Override
	public String toString() {
		// TODO Auto-generated method stub
		return String.valueOf(rl);
	}

}
