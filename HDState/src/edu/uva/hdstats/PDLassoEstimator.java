package edu.uva.hdstats;

public class PDLassoEstimator extends LassoEstimator{
	public PDLassoEstimator(double lambda) {
		super(lambda);
		// TODO Auto-generated constructor stub
	}

	@Override
	public double[][] covariance(double[][] samples) {
		double[][] covar_1=super.covariance(samples);
		return covar_1;
	}

}
