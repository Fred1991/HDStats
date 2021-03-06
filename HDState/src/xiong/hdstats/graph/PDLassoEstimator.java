package xiong.hdstats.graph;

import xiong.hdstats.gaussian.CovarianceEstimator;
import xiong.hdstats.gaussian.LassoEstimator;
import xiong.hdstats.gaussian.NearPD;
import xiong.hdstats.gaussian.SampleCovarianceEstimator;

public class PDLassoEstimator extends SampleCovarianceEstimator{

	private double _lambda;
	private int _iter=CovarianceEstimator.iter;
	
	
	public PDLassoEstimator(double lambda){
		this._lambda=lambda;
	}

	@Override
	public double[][] covariance(double[][] samples) {
		double[][] covar_inner = super.covariance(samples);
		covarianceApprox(covar_inner);
		return covar_inner;
	}
	public double[][] covarianceFromCov(double[][] covar_inner) {
		covarianceApprox(covar_inner);
		return covar_inner;
	}

	
	public void covarianceApprox2(double[][] covar_inner){
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
	
	public void covarianceApprox(double[][] covar_inner){
		System.out.println("starting PDLassoEstimation");
		LassoEstimator le = new LassoEstimator(this._lambda);
		for (int i = 0; i < _iter; i++) {
			le.covarianceApprox(covar_inner);
			NearPD npd = new NearPD();
			System.out.println("starting PD Calculation");
			npd.calcNearPD(new Jama.Matrix(covar_inner));
			System.out.println("finishing PD Calculation");
			double[][] covarx = npd.getX().getArrayCopy();
			for(int k=0;k<covarx.length;k++){
				for(int j=0;j<covarx[k].length;j++){
					covar_inner[k][j]=covarx[k][j];
				}
			}
		}
		System.out.println("finishing PDLassoEstimation");
	}
}
