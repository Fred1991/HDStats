package xiong.hdstats.graph;

import edu.uva.libopt.numeric.NumericFunction;
import edu.uva.libopt.numeric.NumericOptimizer;
import edu.uva.libopt.numeric.Utils;
import edu.uva.libopt.numeric.optimizer.SparseGradientOptimizer;
import xiong.hdstats.gaussian.SampleCovarianceEstimator;


public class DiagKeptLassoEstimator extends SampleCovarianceEstimator{
	private double lambda;
	
	public DiagKeptLassoEstimator(double lambda){
		this.lambda=lambda;
	}
	
	@Override
	public double[][] covariance(double[][] samples) {
		double[][] covar_inner=super.covariance(samples);
		covarianceApprox(covar_inner);
		return covar_inner;
	}
//	@Override
	public void covarianceApprox(double[][] covar_inner){
		double[] covar_2=new double[covar_inner.length*covar_inner.length];
		double[] diag=new double[covar_inner.length];
	//copy diag
		for(int i=0;i<diag.length;i++){
			diag[i]=covar_inner[i][i];
		}
		NumericOptimizer optimizer=new SparseGradientOptimizer(Utils.L1,lambda);
		optimizer.getMinimum(covar_2, stop, 0.00001, new CovarianceDiffNorm1(covar_inner));
		Utils.vec2Matrix(covar_2, covar_inner);
	//	recover diag
		for(int i=0;i<diag.length;i++){
			covar_inner[i][i]=diag[i];
		}
	}
	

	public static class CovarianceDiffNorm1 implements NumericFunction{
		private double[] _org;
		public CovarianceDiffNorm1(double[][] original){
			this._org=new double[original.length*original.length];
			for(int i=0;i<original.length;i++){
				for(int j=0;j<original[i].length;j++){
					if(i!=j)
						_org[i*original.length+j]=original[i][j];
					else
						_org[i*original.length+j]=0;
				}
			}
		}
		@Override
		public double func(double[] X) {
			// TODO Auto-generated method stub
			double error=0;
			for(int i=0;i<X.length;i++){
				error+=(X[i]-_org[i])*(X[i]-_org[i]);
			}			
		//	System.out.println(error);
			return error;
			
			
		}
		
		@Override
		public double[] gradient(double[] X) {
			// TODO Auto-generated method stub
			double[] g=new double[X.length];
			for(int i=0;i<g.length;i++){
				g[i]=2*(X[i]-this._org[i]);
			}
			return g;
		//	return Utils.getGradient(this, X, 0.00001);
		}
		
	}

}
