package xiong.hdstats.gaussian;

import Jama.Matrix;
import xiong.hdstats.da.PseudoInverse;

public abstract class CovarianceEstimator {
	public static double lambda = 0.00001;
	public static int iter = 5;
	public static double stop = 0.0005;

	public abstract double[][] covariance(double[][] samples);

	//public abstract void covarianceApprox(double[][] samples);

	public abstract double[] getMean(double[][] samples);
	
	public double[][] inverseCovariance(double[][] samples){
		return inverse(this.covariance(samples));
	}

	public static double[][] inverse(double[][] _covar) {
		try {
			return new Matrix(_covar).inverse().getArray();
		} catch (Exception e) {
			Matrix inverse = PseudoInverse.inverse(new Matrix(_covar));
		//	if (inverse != null)
				return inverse.getArray();
		//	else {
		//		double[][] im = new double[_covar.length][_covar.length];
		//		for(int i=0;i<im.length;i++)
		//			im[i][i]=1.0;
		//		return im;
		//	}
		}

	}

}
