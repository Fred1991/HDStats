package xiong.hdstats;

import Jama.Matrix;
import xiong.hdstats.da.PseudoInverse;

public abstract class Estimator {
	public static double lambda = 0.00001;
	public static int iter = 5;
	public static double stop = 0.0005;

	public abstract double[][] covariance(double[][] samples);
	
	public abstract void covarianceApprox(double[][] samples);

	public abstract double[] getMean(double[][] samples);

	public static double[][] inverse(double[][] _covar) {
		try {
			return new Matrix(_covar).inverse().getArray();
		} catch (Exception e) {
			return PseudoInverse.inverse(new Matrix(_covar)).getArray();
		}

	}

}
