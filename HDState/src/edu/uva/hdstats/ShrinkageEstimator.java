package edu.uva.hdstats;


public class ShrinkageEstimator extends MLEstimator{

	private double _lambda;
	
	
	public ShrinkageEstimator(double lambda){
		this._lambda=lambda;
	}

	@Override
	public double[][] covariance(double[][] samples) {
		double[][] covar_inner = super.covariance(samples);
		covarianceApprox(covar_inner);
		return covar_inner;
	}

	
	public void covarianceApprox(double[][] covar_inner){
		System.out.println("starting shrinkaging");
		for(int i=0;i<covar_inner.length;i++){
			for(int j=0;j<covar_inner[i].length;j++){
				if(i!=j){
					covar_inner[i][j]=this._lambda*covar_inner[i][j];
				}
			}
		}

		System.out.println("finishing shrinkaging");
	}
}
