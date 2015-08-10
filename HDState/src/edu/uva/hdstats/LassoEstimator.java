package edu.uva.hdstats;

import edu.uva.libopt.numeric.NumericFunction;
import edu.uva.libopt.numeric.NumericOptimizer;
import edu.uva.libopt.numeric.Utils;
import edu.uva.libopt.numeric.optimizer.SparseGradientOptimizer;


public class LassoEstimator extends LDEstimator{
	private double lambda;
	
	public LassoEstimator(double lambda){
		this.lambda=lambda;
	}
	
	@Override
	public double[][] covariance(double[][] samples) {
		double[][] covar_1=super.covariance(samples);
		double[] covar_2=new double[covar_1.length*covar_1.length];
		NumericOptimizer optimizer=new SparseGradientOptimizer(Utils.L1,lambda);
		optimizer.getMinimum(covar_2, 0.001, 0.0001, new CovarianceDiffNorm1(covar_1));
		double[][] covariance=new double[covar_1.length][covar_1.length];
		Utils.vec2Matrix(covar_2, covariance);
		return covariance;
	}
	
	public static class CovarianceDiffNorm1 implements NumericFunction{
		private double[] _org;
		public CovarianceDiffNorm1(double[][] original){
			this._org=new double[original.length*original.length];
			for(int i=0;i<original.length;i++){
				for(int j=0;j<original[i].length;j++){
					_org[i*original.length+j]=original[i][j];
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
			return Math.pow(error, 0.5);
			
			
		}
		
	}

}
