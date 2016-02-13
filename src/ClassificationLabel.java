package cs362;

import java.io.Serializable;

public class ClassificationLabel extends Label implements Serializable {
	public int cl;
	public ClassificationLabel(int label) {
		// TODO Auto-generated constructor stub
		cl = label;
	}

	@Override
	public String toString() {
		// TODO Auto-generated method stub
		return String.valueOf(cl);
	}

}
