package edu.uva.hdstats;


public class PDLassoEstimator extends LassoEstimator{
	private double[][] covar_inner;

	public PDLassoEstimator(double lambda) {
		super(lambda);
	}

	@Override
	public double[][] covariance(double[][] samples) {
		covar_inner = super.covariance(samples);
		NearPD npd=new NearPD();
		npd.calcNearPD(new Jama.Matrix(this.covar_inner));
		return npd.getX().getArrayCopy();
	}
}
