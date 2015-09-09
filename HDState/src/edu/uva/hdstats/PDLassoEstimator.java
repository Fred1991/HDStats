package edu.uva.hdstats;


public class PDLassoEstimator extends LassoEstimator{

	public PDLassoEstimator(double lambda) {
		super(lambda);
	}

	@Override
	public double[][] covariance(double[][] samples) {
		covar_inner = super.covariance(samples);
		return npdApprox();
	}

	public double[][] npdApprox() {
		NearPD npd=new NearPD();
		npd.calcNearPD(new Jama.Matrix(this.covar_inner));
		return npd.getX().getArrayCopy();
	}
}
