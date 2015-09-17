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
		covarianceApprox(covar_inner);
		return covar_inner;
	}

	
	@Override
	public void covarianceApprox(double[][] covar_inner){
		LassoEstimator le = new LassoEstimator(this._lambda);
		for (int i = 0; i < _iter; i++) {
			le.covarianceApprox(covar_inner);
			NearPD npd = new NearPD();
			npd.calcNearPD(new Jama.Matrix(covar_inner));
			double[][] covarx = npd.getX().getArrayCopy();
			for(int k=0;k<covarx.length;k++){
				for(int j=0;j<covarx[k].length;j++){
					covar_inner[k][j]=covarx[k][j];
				}
			}
		}
	}
	

}
