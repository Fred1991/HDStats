package edu.uva.hdstats.da;

import ml.classification.LogisticRegression;

public class LRClassifier implements Classifier<double[]>{
	LogisticRegression classifier=new LogisticRegression(1,0.1);
	
	public LRClassifier(double[][] d, int[] g) {
		classifier.feedData(d);
		classifier.feedLabels(g);
		classifier.train();

	}

	
	@Override
	public int predict(double[] x) {
		// TODO Auto-generated method stub		
		return classifier.predict(new double[][]{x})[0];
	}

	@Override
	public int predict(double[] x, double[] posteriori) {
		// TODO Auto-generated method stub
		return this.predict(x);
	}

}
