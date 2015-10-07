package edu.uva.hdstats;

public abstract class Estimator {
	public static double lambda = 0.00001;
	public static int iter = 1;
	public static double stop=0.005;
	
	public abstract double[][] covariance(double[][] samples);
	public abstract void covarianceApprox(double[][] samples);
	public abstract double[] getMean(double[][] samples);

}
