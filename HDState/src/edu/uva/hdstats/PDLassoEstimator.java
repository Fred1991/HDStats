package edu.uva.hdstats;


public class PDLassoEstimator extends LDEstimator{

	private double _lambda;
	
	public PDLassoEstimator(double lambda){
		this._lambda=lambda;
	}

	@Override
	public double[][] covariance(double[][] samples) {
		double[][] covar_inner = super.covariance(samples);
		covarianceApprox(covar_inner);
		return covar_inner;
	}

	@Override
	public void covarianceApprox(double[][] covar_inner){
		LassoEstimator le=new LassoEstimator(this._lambda);
		le.covarianceApprox(covar_inner);
		NearPD npd=new NearPD();
		npd.calcNearPD(new Jama.Matrix(covar_inner));
		double[][] covarx=npd.getX().getArrayCopy();
		for(int i=0;i<covarx.length;i++){
			for(int j=0;j<covarx[i].length;j++){
				covar_inner[i][j]=covarx[i][j];
			}
		}
	}
	
}
