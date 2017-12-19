package xiong.hdstats.mat;

import Jama.Matrix;
import Jama.QRDecomposition;
import edu.uva.libopt.numeric.Utils;
import smile.stat.distribution.SpikedMultivariateGaussianDistribution;
import xiong.hdstats.gaussian.SampleCovarianceEstimator;

public class RandSVD {

	public static Matrix[] decompose(Matrix A, int k, int p, boolean top) {
		Matrix[] mtx = new Matrix[3];
		Matrix omega = Matrix.random(A.getColumnDimension(), k + p);
		System.out.println(omega.getRowDimension() + "\t" + omega.getColumnDimension() + "\t times\t"
				+ A.getRowDimension() + "\t" + A.getColumnDimension());
		Matrix Y = A.times(omega);
		QRDecomposition qrd = Y.qr();
		Matrix Q = qrd.getQ();
		System.out.println(Q.getRowDimension() + "\t" + Q.getColumnDimension() + "\t times\t" + A.getRowDimension()
				+ "\t" + A.getColumnDimension());

		Matrix B = Q.transpose().times(A);
		System.out.println(B.getRowDimension() + "\t" + B.getColumnDimension());
		Matrix[] tsvd = TruncatedSVD.decomposeScale(B, k, top);
		mtx[1] = tsvd[1].copy();
		mtx[2] = tsvd[2].copy();
		return mtx;
	}

	public static Matrix spikedCovarianceMatrix(Matrix data, int k, int p) {
		Matrix[] USV = decompose(data, k, p, true);
		Matrix S = USV[1];
		Matrix V = USV[2];
		return V.times(S.times(S.transpose())).times(V.transpose()).times(1.0 / data.getRowDimension());
	}

	public static Matrix spikedInverseCovarianceMatrix(Matrix data, int k, int p) {
		Matrix[] USV = decompose(data, k, p, false);
		Matrix S = USV[1];
		Matrix V = USV[2];
		return V.times(S.times(S.transpose()).inverse()).times(V.transpose()).times(1.0 * data.getRowDimension());
	}

	public static double[][] spikedCovarianceMatrix(double[][] data, int k, int p) {
		Matrix spikedCov = null;
		if (data.length >= data[0].length) {
			spikedCov = spikedCovarianceMatrix(new Matrix(data), k, p);
		} else {
			double[][] _data = new double[data[0].length][data[0].length];
			for (int i = 0; i < _data.length; i++) {
				for (int j = 0; j < _data[i].length; j++) {
					_data[i][j] = data[Math.abs(i % data.length)][j];
				}
			}
			spikedCov = spikedCovarianceMatrix(new Matrix(_data), k, p);
		}
		return spikedCov.getArrayCopy();
	}

	
	public static double[][] spikedInverseCovarianceMatrix(double[][] data, int k, int p) {
		Matrix spikedICov = null;
		if (data.length >= data[0].length) {
			spikedICov = spikedInverseCovarianceMatrix(new Matrix(data), k, p);
		} else {
			double[][] _data = new double[data[0].length][data[0].length];
			for (int i = 0; i < _data.length; i++) {
				for (int j = 0; j < _data[i].length; j++) {
					_data[i][j] = data[Math.abs(i % data.length)][j];
				}
			}
			spikedICov = spikedInverseCovarianceMatrix(new Matrix(_data), k, p);
		}
		return spikedICov.getArrayCopy();
	}

	public static void main(String[] args) {
		int p = 200;
		int s = 10;
		int n = 100;

		double[][] cov = new double[p][p];
		for (int i = 0; i < cov.length; i++) {
			for (int j = 0; j < cov.length; j++) {
				cov[i][j] = Math.pow(0.6, Math.abs(i - j));
			}
		}

		double[][] ocov = TruncatedSVD.truncate(cov, s);
		double[][] icov = TruncatedSVD.truncatedInverse(cov, s);

		double[] zero = new double[p];
		SpikedMultivariateGaussianDistribution gen = new SpikedMultivariateGaussianDistribution(zero, ocov);

		double[][] data = new double[n][p];
		for (int i = 0; i < n; i++) {
			double[] vecp = gen.rand();
			for (int j = 0; j < p; j++) {
				data[i][j] = vecp[j];
			}
		}
		double[][] estICov = RandSVD.spikedInverseCovarianceMatrix(data, s, s);
		double[][] samICov = TruncatedSVD.truncate(new SampleCovarianceEstimator().inverseCovariance(data), s);

		System.out.println("true\t" + new Matrix(icov).normF());
		System.out.println("estimated\t" + new Matrix(estICov).normF());
		System.out.println("sample\t" + new Matrix(samICov).normF());
		System.out.println(samICov.length + "\t" + samICov[0].length + "\t" + Utils.getErrorL2(icov, estICov) + "\t"
				+ Utils.getErrorL2(icov, samICov));
	}
}
