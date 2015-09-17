package edu.uva.hdstats;

public interface Estimator {
	public static final double lambda = 0.02;
	public static final int iter = 5;
	
	public double[][] covariance(double[][] samples);
	public void covarianceApprox(double[][] samples);
	public double[] getMean(double[][] samples);

}
