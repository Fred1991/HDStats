package edu.uva.hdstats;

public class IterPDLassoEstimator extends LDEstimator {

	public int _iter;
	public double _lambda;

	public IterPDLassoEstimator(double lambda, int iter) {
		this._lambda = lambda;
		this._iter = iter;
	}

	@Override
	public double[][] covariance(double[][] samples) {
		double[][] covar_inner = super.covariance(samples);
		return covarianceApprox(covar_inner);
	}

	@Override
	public double[][] covarianceApprox(double[][] covar_inner) {
		LassoEstimator le = new LassoEstimator(this._lambda);
		for (int i = 0; i < _iter; i++) {
			covar_inner = le.covarianceApprox(covar_inner);
			NearPD npd = new NearPD();
			npd.calcNearPD(new Jama.Matrix(covar_inner));
			covar_inner = npd.getX().getArrayCopy();
		}
		return covar_inner;
	}
}
