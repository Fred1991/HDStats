package edu.uva.hdstats.da;
import smile.classification.*;

public class DTreeClassifier implements Classifier<double[]>{
	DecisionTree classifier = null;
	
	public DTreeClassifier(double[][] d, int[] g, int k) {
		classifier=new DecisionTree(d,g,k);
	}

	
	@Override
	public int predict(double[] x) {
		// TODO Auto-generated method stub
		return classifier.predict(x);
	}

	@Override
	public int predict(double[] x, double[] posteriori) {
		// TODO Auto-generated method stub
		return this.predict(x);
	}

}
