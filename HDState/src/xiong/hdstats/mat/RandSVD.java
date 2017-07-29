package xiong.hdstats.mat;

import Jama.Matrix;
import Jama.QRDecomposition;
import edu.uva.libopt.numeric.Utils;
import smile.stat.distribution.GLassoMultivariateGaussianDistribution;
import xiong.hdstats.MLEstimator;

public class RandSVD {
	public static int numPositiveEigs(Matrix D) {
		for (int i = 0; i < D.getColumnDimension(); i++) {
	//		System.out.println(D.get(i, i));
			if (D.get(i, i) < 0) {
				return i;
			}
		}
		return D.getColumnDimension();
	}

	public static Matrix[] decompose(Matrix A, int k, int p) {
		Matrix[] mtx = new Matrix[3];
		Matrix omega = Matrix.random(A.getRowDimension(), k+p);
		Matrix Y = A.times(omega);
		QRDecomposition qrd =Y.qr();
		Matrix Q = qrd.getQ();
		Matrix B = Q.transpose().times(A);
		System.out.println(B.getRowDimension()+"\t"+B.getColumnDimension());
		Matrix[] tsvd = TruncatedSVD.decomposeScale(B, k);
			// numPositiveEigs(S);

	//	mtx[0] = Q.times(tsvd[0]).copy();
		mtx[1] = tsvd[1].copy();
		mtx[2] = tsvd[2].copy();
		return mtx;
	}

	public static Matrix spikedCovarianceMatrix(Matrix data, int k, int p) {
		Matrix[] USV = decompose(data, k, p);
		Matrix S = USV[1];
		Matrix V = USV[2];
		// System.out.println(data.getRowDimension());
		return V.times(S).times(S.transpose()).times(V.transpose()).times(1.0 / data.getRowDimension());
	}

	public static double[][] spikedCovarianceMatrix(double[][] data, int k, int p, double lambda) {
		Matrix ident = 	Matrix.identity(data[0].length, data[0].length);
		Matrix spikedCov = null;
		if (data.length >= data[0].length) {
			spikedCov= spikedCovarianceMatrix(new Matrix(data), k, p);
		} else {
			double[][] _data = new double[data[0].length][data[0].length];
			for (int i = 0; i < _data.length; i++) {
				for (int j = 0; j < _data[i].length; j++) {
					_data[i][j] = data[Math.abs(i % data.length)][j];
				}
			}
			spikedCov= spikedCovarianceMatrix(new Matrix(_data), k, p);
		}
		return spikedCov.plus(ident.times(lambda)).getArrayCopy();
	}

	public static void main(String[] args) {
		int p = 1000;
		int n = 10;
		double[][] cov = new double[p][p];
		for (int i = 0; i < cov.length; i++) {
			for (int j = 0; j < cov.length; j++) {
				cov[i][j] = Math.pow(0.6, Math.abs(i - j));
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
		double[][] estCov = RandSVD.spikedCovarianceMatrix(data, 1, 1, 0.5);
		double[][] samCov = new MLEstimator().covariance(data);
		// double[][] samCov = new Matrix(data).transpose().times(new
		// Matrix(data)).times(1.0/n).getArray();
		System.out.println(samCov.length + "\t" + samCov[0].length + "\t" + Utils.getErrorL1(cov, estCov) + "\t"
				+ Utils.getErrorL1(cov, samCov));
	}
}
