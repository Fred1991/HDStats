package xiong.hdstats.gaussian;

import edu.uva.libopt.numeric.Utils;
import la.decomposition.LUDecomposition;
import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import xiong.hdstats.ProbabilisticModel;

public class GaussianModel implements ProbabilisticModel {
	private Matrix icovariance;
	private double l1N_icovar=0;
	private Matrix mean;
	

	@Override
	public void buildModel(double[][] covariance, double[] mean) {
		// TODO Auto-generated method stub
		this.icovariance=new LUDecomposition(new DenseMatrix(covariance)).inverse();
		this.mean=new DenseMatrix(mean,mean.length);
		for(double[] covs:covariance)
			for(double cov:covs)
				this.l1N_icovar+=Math.abs(cov);
	}

	@Override
	public double getProbability(double[] values) {
		// TODO Auto-generated method stub
		
		Matrix valuex = new DenseMatrix(values, values.length);
		Matrix errorX = valuex.minus(mean);
		Matrix errorT = errorX.transpose();
		Matrix product = (errorT.times(icovariance)).times(errorX);
		double up = -0.5 * product.getEntry(0, 0);
		return 1.0 / (Math.pow((2 * Math.PI), values.length / 2.0) * this.l1N_icovar)
				* Math.exp(up);

	}
}
