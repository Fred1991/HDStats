package xiong.hdstats.mat;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import edu.uva.libopt.numeric.Utils;
import smile.stat.distribution.SpikedMultivariateGaussianDistribution;
import xiong.hdstats.gaussian.CovarianceEstimator;
import xiong.hdstats.gaussian.DBGLassoEstimator;
import xiong.hdstats.gaussian.GLassoEstimator;
import xiong.hdstats.gaussian.SampleCovarianceEstimator;

public class TruncatedEVD {

	public static int numPositiveEigs(Matrix D) {
		for (int i = D.getColumnDimension() - 1; i >= 0; i--) {
			// System.out.println(D.get(i, i));
			if (D.get(i, i) < 0) {
				return i;
			}
		}
		return D.getColumnDimension();
	}

	public static Matrix[] decompose(Matrix A, int k) {
		Matrix[] mtx = new Matrix[2];
		EigenvalueDecomposition evd = A.eig();
		Matrix D = evd.getD();
		int npe = numPositiveEigs(D);
		k = Math.min(k, npe);
		// System.out.println(npe);
		Matrix VT = evd.getV().transpose();
		mtx[0] = D.getMatrix(D.getRowDimension() - k, D.getRowDimension() - 1, D.getRowDimension() - k,
				D.getRowDimension() - 1).copy();

		numPositiveEigs(mtx[0]);

		// mtx[1] = VT.transpose();

		mtx[1] = VT.getMatrix(D.getRowDimension() - k, D.getRowDimension() - 1, 0, VT.getColumnDimension() - 1).copy()
				.transpose();
		return mtx;
	}

	public static Matrix spikedCovarianceMatrix(Matrix data, int k) {
		Matrix[] DV = decompose(data, k);
		Matrix D = DV[0];
		Matrix V = DV[1];
		// System.out.println(data.getRowDimension());
		return V.times(D).times(V.transpose());
	}

	public static Matrix spikedInverseCovarianceMatrix(Matrix data, int k) {
		Matrix[] DV = decompose(data, k);
		Matrix DI = DV[0].inverse();
		Matrix V = DV[1];
		// System.out.println(data.getRowDimension());
		return V.times(DI).times(V.transpose());
	}

	public static double[][] spikedCovarianceMatrix(double[][] data, int k, double lambda) {
		Matrix ident = Matrix.identity(data[0].length, data[0].length);
		Matrix sampleCov = new Matrix(new SampleCovarianceEstimator().covariance(data));
		Matrix spikedCov = spikedCovarianceMatrix(sampleCov, k);
		return spikedCov.plus(ident.times(lambda)).copy().getArray();
	}

	public static void main(String[] args) {
		int p = 1000;
		int n = 50;
		double[][] cov = new double[p][p];
		for (int i = 0; i < cov.length; i++) {
			for (int j = 0; j < cov.length; j++) {
				cov[i][j] = Math.pow(0.8, Math.abs(i - j));
			}
		}
		double[] zero = new double[p];
		SpikedMultivariateGaussianDistribution gen = new SpikedMultivariateGaussianDistribution(zero, cov);

		double[][] data = new double[n][p];
		for (int i = 0; i < n; i++) {
			double[] vecp = gen.rand();
			for (int j = 0; j < p; j++) {
				data[i][j] = vecp[j];
			}
		}
		Matrix ident = Matrix.identity(p, p);
		double lambda = 1.0;
		int k = 5;
		int d = k;

		long start = System.currentTimeMillis();

		double[][] samCov = new SampleCovarianceEstimator().covariance(data);
		long t0 = System.currentTimeMillis();

		double[][] estCov = TruncatedEVD.spikedCovarianceMatrix(data, k, 0.0);
		long t1 = System.currentTimeMillis();

		double[][] estCov2 = TruncatedSVD.spikedCovarianceMatrix(data, k);
		long t2 = System.currentTimeMillis();

		double[][] estCov3 = RandSVD.spikedInverseCovarianceMatrix(data, k, d, true);
		long t3 = System.currentTimeMillis();

		double[][] gLasso = new GLassoEstimator(12)._glassoPrecisionMatrix(samCov);
		long t4 = System.currentTimeMillis();

		double[][] dgLasso = new DBGLassoEstimator(12)._deSparsifiedGlassoPrecisionMatrix(samCov);
		long t5 = System.currentTimeMillis();

		double[][] graph = new Matrix(cov).plus(ident.times(lambda)).inverse().getArray();
		double[][] graphEst = new Matrix(estCov).plus(ident.times(lambda)).inverse().getArray();
		double[][] graphEst2 = new Matrix(estCov2).plus(ident.times(lambda)).inverse().getArray();
		double[][] graphEst3 = new Matrix(estCov3).plus(ident.times(lambda)).inverse().getArray();

		double[][] graphSam = CovarianceEstimator.inverse(new Matrix(samCov).getArray());
		CovarianceEstimator.lambda = 16.0;
		double[][] gCov = SampleCovarianceEstimator.inverse(gLasso);
		CovarianceEstimator.lambda = 16.0;

		double[][] dgCov = SampleCovarianceEstimator.inverse(dgLasso);

		// double[][] samCov = new Matrix(data).transpose().times(new
		// Matrix(data)).times(1.0/n).getArray();
		System.out
				.println("Truncated-EVD\t" + "Truncated-SVD\t" + "TRand-SVD\t" + "GLasso\t" + "DB-GLasso\t" + "Sample");
		System.out.println("" + Utils.getErrorL2(cov, estCov) + "\t" + Utils.getErrorL2(cov, estCov2) + "\t"
				+ Utils.getErrorL2(cov, estCov3) + "\t" + Utils.getErrorL2(cov, gCov) + "\t"
				+ Utils.getErrorL2(cov, dgCov) + "\t" + Utils.getErrorL2(cov, samCov) + "\n"

				+ Utils.getErrorL2(graph, graphEst) + "\t" + Utils.getErrorL2(graph, graphEst2) + "\t"
				+ Utils.getErrorL2(graph, graphEst3) + "\t" + Utils.getErrorL2(graph, gLasso) + "\t"
				+ Utils.getErrorL2(graph, dgLasso) + "\t" + Utils.getErrorL2(graph, graphSam) + "\n"

				+ (t1 - t0) + "\t" + (t2 - t1) + "\t" + (t3 - t2) + "\t" + (t4 - t3) + "\t" + (t5 - t4) + "\t"
				+ (t0 - start));
	}

}
