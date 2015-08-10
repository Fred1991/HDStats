package edu.uva.hdstats;

public interface Estimator {
	
	public double[][] covariance(double[][] samples);
	public double[] getMean(double[][] samples);

}
