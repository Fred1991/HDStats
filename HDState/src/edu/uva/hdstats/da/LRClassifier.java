package edu.uva.hdstats.da;

import ml.classification.LogisticRegression;

public class LRClassifier implements Classifier<double[]>{
	LogisticRegression classifier;
	
	public LRClassifier(double[][] d, int[] g, double l) {
		this.classifier=new LogisticRegression(1,l);
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
