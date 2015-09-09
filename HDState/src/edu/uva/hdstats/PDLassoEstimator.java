package edu.uva.hdstats;


public class PDLassoEstimator extends LDEstimator{

	private double _lambda;
	
	public PDLassoEstimator(double lambda){
		this._lambda=lambda;
	}

	@Override
	public double[][] covariance(double[][] samples) {
		double[][] covar_inner = super.covariance(samples);
		return covarianceApprox(covar_inner);
	}

	@Override
	public double[][] covarianceApprox(double[][] covar_inner){
		LassoEstimator le=new LassoEstimator(this._lambda);
		covar_inner=le.covarianceApprox(covar_inner);
		NearPD npd=new NearPD();
		npd.calcNearPD(new Jama.Matrix(covar_inner));
		return npd.getX().getArrayCopy();
	}
}
