package xiong.hdstats.mat;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import Jama.SingularValueDecomposition;
import edu.uva.libopt.numeric.Utils;
import smile.stat.distribution.SpikedMultivariateGaussianDistribution;
import xiong.hdstats.gaussian.SampleCovarianceEstimator;

public class TruncatedSVD {

	private static int numberOfNonZeroSVs(Matrix S) {
		int count = 0;
		for (int i = 0; i < S.getColumnDimension(); i++) {
	//		System.out.println("the no. " + i + " value:\t" + S.get(i, i));
			if (S.get(i, i) != 0 && Math.abs(S.get(i, i)) >= 1.0e-5)
				count++;
		}
		return count;
	}

	private static Matrix[] decompose(Matrix A, int k, boolean top) {
		Matrix[] mtx = new Matrix[3];
		SingularValueDecomposition svd = A.svd();
		Matrix U = svd.getU();
		Matrix S = svd.getS();
		Matrix VT = svd.getV().transpose();
		int nz = numberOfNonZeroSVs(S);

		if (nz < k)
			k = nz;

		if (!top) {
			mtx[0] = U.getMatrix(0, U.getRowDimension() - 1, nz - k, nz - 1).copy();
			mtx[1] = S.getMatrix(nz - k, nz - 1, nz - k, nz - 1).copy();
			mtx[2] = VT.getMatrix(nz - k, nz - 1, 0, VT.getColumnDimension() - 1).copy().transpose();
		} else {
			mtx[0] = U.getMatrix(0, U.getRowDimension() - 1, 0, k - 1).copy();
			mtx[1] = S.getMatrix(0, k - 1, 0, k - 1).copy();
			mtx[2] = VT.getMatrix(0, k - 1, 0, VT.getColumnDimension() - 1).copy().transpose();
		}
		return mtx;
	}

	public static Matrix[] decomposeScale(Matrix A, int k, boolean top) {
		double[][] data = A.getArray();
		if (data.length >= data[0].length) {
			return decompose(A, k, top);
		} else {
			double[][] _data = new double[data[0].length][data[0].length];
			for (int i = 0; i < _data.length; i++) {
				if (i < data.length) {
					for (int j = 0; j < _data[i].length; j++) {
						_data[i][j] = data[i][j];
					}
				}
			}
			return decompose(new Matrix(_data), k, top);
		}
	}

	public static Matrix spikedCovarianceMatrix(Matrix data, int k) {
		Matrix[] USV = decomposeScale(data, k, true);
		Matrix S = USV[1];
		Matrix V = USV[2];
		// System.out.println(data.getRowDimension());
		return (V.times(S.times(S.transpose().times(V.transpose())))).times(1.0 / data.getRowDimension());
	}

	public static Matrix spikedInverseCovarianceMatrix(Matrix data, int k) {
		Matrix[] USV = decomposeScale(data, k, false);
		for (int i = 0; i < USV[1].getRowDimension(); i++)
			System.out.println("S\t" + USV[1].get(i, i));
		Matrix SI = USV[1].inverse();
		Matrix V = USV[2];
		// System.out.println(data.getRowDimension());
		return (V.times(SI.times(SI.transpose().times(V.transpose())))).times(1.0 * data.getRowDimension());
	}

	public static double[][] spikedCovarianceMatrix(double[][] data, int k) {
		Matrix spikedCov = spikedCovarianceMatrix(new Matrix(data), k);
		return spikedCov.getArrayCopy();
	}

	public static double[][] spikedInverseCovarianceMatrix(double[][] data, int k) {
		Matrix spikedCov = spikedInverseCovarianceMatrix(new Matrix(data), k);
		return spikedCov.getArrayCopy();
	}

	public static double[][] eigenTruncate(double[][] cov, int s) {
		Matrix covMat = new Matrix(cov);
		EigenvalueDecomposition evd = covMat.eig();
		Matrix V = evd.getV();
		Matrix D = evd.getD();
		Matrix SI = new Matrix(D.getRowDimension(), D.getColumnDimension());
		for (int i = 0; i < D.getRowDimension(); i++) {
			if (i < D.getRowDimension() - s) {
				D.set(i, i, 0);
				SI.set(i, i, 0);
			} else {
				SI.set(i, i, 1.0 / D.get(i, i));
			}
		}
		covMat = V.times(D.times(V.transpose()));
		return covMat.getArrayCopy();
	}

	public static double[][] eigenTruncatedInverse(double[][] cov, int s) {
		Matrix covMat = new Matrix(cov);
		EigenvalueDecomposition evd = covMat.eig();
		Matrix V = evd.getV();
		Matrix D = evd.getD();
		Matrix SI = new Matrix(D.getRowDimension(), D.getColumnDimension());
		for (int i = 0; i < D.getRowDimension(); i++) {
			if (i < D.getRowDimension() - s) {
				D.set(i, i, 0);
				SI.set(i, i, 0);
			} else {
				SI.set(i, i, 1.0 / D.get(i, i));
			}
		}
		Matrix iCovMat = V.times(SI.times(V.transpose()));
		return iCovMat.getArrayCopy();
	}

	public static void main(String[] args) {
		int p = 200;
		int s = 10;
		int n = 50;

		double[][] cov = new double[p][p];
		for (int i = 0; i < cov.length; i++) {
			for (int j = 0; j < cov.length; j++) {
				cov[i][j] = Math.pow(0.6, Math.abs(i - j));
			}
		}

		double[][] ocov = eigenTruncate(cov, s);
		double[][] icov = eigenTruncatedInverse(cov, s);

		double[] zero = new double[p];
		SpikedMultivariateGaussianDistribution gen = new SpikedMultivariateGaussianDistribution(zero, ocov);

		double[][] data = new double[n][p];
		for (int i = 0; i < n; i++) {
			double[] vecp = gen.rand();
			for (int j = 0; j < p; j++) {
				data[i][j] = vecp[j];
			}
		}
		double[][] estICov = TruncatedSVD.spikedInverseCovarianceMatrix(data, s);
		double[][] samICov = eigenTruncate(new SampleCovarianceEstimator().inverseCovariance(data), s);

		System.out.println("true\t" + new Matrix(icov).normF());
		System.out.println("estimated\t" + new Matrix(estICov).normF());
		System.out.println("sample\t" + new Matrix(samICov).normF());
		System.out.println(samICov.length + "\t" + samICov[0].length + "\t" + Utils.getErrorL2(icov, estICov) + "\t"
				+ Utils.getErrorL2(icov, samICov));
	}

}
