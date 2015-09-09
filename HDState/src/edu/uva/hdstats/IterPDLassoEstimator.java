package edu.uva.hdstats;


public class IterPDLassoEstimator extends PDLassoEstimator{

	public IterPDLassoEstimator(double lambda) {
		super(lambda);
	}

	@Override
	public double[][] covariance(double[][] samples) {
		covar_inner = super.covariance(samples);
		covar_inner =  npdApprox();
		covar_inner = super.covariance(samples);
		covar_inner =  npdApprox();
		covar_inner = super.covariance(samples);
		covar_inner =  npdApprox();
		covar_inner = super.covariance(samples);
		covar_inner =  npdApprox();
		covar_inner = super.covariance(samples);
		covar_inner =  npdApprox();
		return covar_inner;
	}

}
