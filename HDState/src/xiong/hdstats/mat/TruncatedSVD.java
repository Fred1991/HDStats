package xiong.hdstats.mat;

import Jama.Matrix;
import Jama.SingularValueDecomposition;
import edu.uva.libopt.numeric.Utils;
import smile.stat.distribution.GLassoMultivariateGaussianDistribution;
import xiong.hdstats.MLEstimator;

public class TruncatedSVD {

	public static int numPositiveEigs(Matrix D) {
		for (int i = 0; i < D.getColumnDimension(); i++) {
			// System.out.println(D.get(i, i));
			if (D.get(i, i) < 0) {
				return i;
			}
		}
		return D.getColumnDimension();
	}

	private static Matrix[] decompose(Matrix A, int k) {
		Matrix[] mtx = new Matrix[3];
		SingularValueDecomposition svd = A.svd();
		Matrix U = svd.getU();
		Matrix S = svd.getS();
		Matrix VT = svd.getV().transpose();

		// numPositiveEigs(S);

		mtx[0] = U.getMatrix(0, U.getRowDimension() - 1, 0, k - 1).copy();
		mtx[1] = S.getMatrix(0, k - 1, 0, k - 1).copy();
		mtx[2] = VT.getMatrix(0, k - 1, 0, VT.getColumnDimension() - 1).copy().transpose();
		return mtx;
	}

	public static Matrix[] decomposeScale(Matrix A, int k) {
		double[][] data = A.getArray();
		if (data.length >= data[0].length) {
			return decompose(A, k);
		} else {
			double[][] _data = new double[data[0].length][data[0].length];
			for (int i = 0; i < _data.length; i++) {
				if (i < data.length) {
					for (int j = 0; j < _data[i].length; j++) {
						_data[i][j] = data[i][j];
					}
				}
			}
			return decompose(new Matrix(_data), k);
		}
	}

	public static Matrix spikedCovarianceMatrix(Matrix data, int k) {
		Matrix[] USV = decomposeScale(data, k);
		Matrix S = USV[1];
		Matrix V = USV[2];
		// System.out.println(data.getRowDimension());
		return V.times(S).times(S.transpose()).times(V.transpose()).times(1.0 / data.getRowDimension());
	}

	public static double[][] spikedCovarianceMatrix(double[][] data, int k, double lambda) {
		Matrix ident = Matrix.identity(data[0].length, data[0].length);
		Matrix spikedCov = spikedCovarianceMatrix(new Matrix(data), k);
		return spikedCov.plus(ident.times(lambda)).getArrayCopy();
	}

	public static void main(String[] args) {
		int p = 200;
		int n = 10;
		double[][] cov = new double[p][p];
		for (int i = 0; i < cov.length; i++) {
			for (int j = 0; j < cov.length; j++) {
				cov[i][j] = Math.pow(0.8, Math.abs(i - j) * 5);
			}
		}
		double[] zero = new double[p];
		GLassoMultivariateGaussianDistribution gen = new GLassoMultivariateGaussianDistribution(zero, cov);

		double[][] data = new double[n][p];
		for (int i = 0; i < n; i++) {
			double[] vecp = gen.rand();
			for (int j = 0; j < p; j++) {
				data[i][j] = vecp[j];
			}
		}
		double[][] estCov = TruncatedSVD.spikedCovarianceMatrix(data, 1, 0.0);
		double[][] samCov = new MLEstimator().covariance(data);
		// double[][] samCov = new Matrix(data).transpose().times(new
		// Matrix(data)).times(1.0/n).getArray();
		System.out.println(samCov.length + "\t" + samCov[0].length + "\t" + Utils.getErrorL1(cov, estCov) + "\t"
				+ Utils.getErrorL1(cov, samCov));
	}

}
