package xiong.hdstats.graph.ens;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import gov.sandia.cognition.math.RingAccumulator;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.matrix.mtj.DenseMatrixFactoryMTJ;
import gov.sandia.cognition.math.matrix.mtj.decomposition.CholeskyDecompositionMTJ;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import xiong.hdstats.Estimator;
import xiong.hdstats.MLEstimator;
import xiong.hdstats.NearPD;
import xiong.hdstats.gaussian.NonSparseEstimator;
import xiong.hdstats.graph.DGLassoGraph;

public class WishartDGLassoGraph extends MLEstimator {

	public double[][] wishartMeanPrecision;
	public double[] mean;
	public List<DGLassoGraph> sampledGraphs = new ArrayList<DGLassoGraph>();
	public int size;
	private NonSparseEstimator ne = new NonSparseEstimator();

	public WishartDGLassoGraph(double[][] data, double lambda, int size, double K) {
		Estimator.lambda = lambda;
		this.size = size;
		this.covariance(data);
		this.mean = this.getMean(data);

		for (int i = 0; i < this.wishartMeanPrecision.length; i++)
			this.wishartMeanPrecision[i][i] += 1;

		VectorFactory vf = VectorFactory.getDenseDefault();
		MatrixFactory mf = MatrixFactory.getDenseDefault();
		Vector meanV = vf.copyArray(mean);
		Matrix covarianceSqrt = null;
		try {
			covarianceSqrt = CholeskyDecompositionMTJ
					.create(DenseMatrixFactoryMTJ.INSTANCE.copyMatrix(mf.copyArray(wishartMeanPrecision).inverse()))
					.getR();
		} catch (Exception exp) {
			NearPD npd = new NearPD();
			npd.calcNearPD(new Jama.Matrix(mf.copyArray(wishartMeanPrecision).inverse().toArray()));
			covarianceSqrt = CholeskyDecompositionMTJ
					.create(DenseMatrixFactoryMTJ.INSTANCE.copyMatrix(mf.copyArray(npd.getX().getArray())))
					.getR();
		}
		int fDOF = Math.min(Math.max(this.wishartMeanPrecision.length, data.length) * 10, 2000);
		Random r = new Random(System.currentTimeMillis());
		while (sampledGraphs.size() < this.size) {
			Matrix mtx = sample(r, meanV, covarianceSqrt, fDOF);
			sampledGraphs.add(new DGLassoGraph(mtx.inverse().toArray(), lambda));
		}
	}

	@Override
	public double[][] covariance(double[][] samples) {
		this.wishartMeanPrecision = ne._deSparsifiedGlassoPrecisionMatrix(super.covariance(samples));
		return inverse(this.wishartMeanPrecision);
	}

	public int[][] thresholding(double t, int e) {
		List<int[][]> tGraphs = new ArrayList<int[][]>();
		for (int i = 0; i < this.size; i++)
			tGraphs.add(sampledGraphs.get(i).thresholding(t));
		return this.submodularSearch(tGraphs, e);
	}

	public int[][] adaptiveThresholding(double t, int e) {
		List<int[][]> tGraphs = new ArrayList<int[][]>();
		for (int i = 0; i < this.size; i++)
			tGraphs.add(sampledGraphs.get(i).adaptiveThresholding(t));
		return this.submodularSearch(tGraphs, e);
	}

	public int[][] thresholdingDiff(double t, double e, WishartDGLassoGraph wg) {
		List<int[][]> tGraphs = new ArrayList<int[][]>();
		for (int i = 0; i < this.size; i++)
			tGraphs.add(sampledGraphs.get(i).thresholding(t));
		List<int[][]> wGraphs = new ArrayList<int[][]>();
		for (int i = 0; i < this.size; i++)
			wGraphs.add(wg.sampledGraphs.get(i).thresholding(t));

		return this.submodularSubmodularSearch(tGraphs, wGraphs, e);
	}

	public int[][] adaptiveThresholdingDiff(double t, double overlap, WishartDGLassoGraph wg) {
		List<int[][]> tGraphs = new ArrayList<int[][]>();
		for (int i = 0; i < this.size; i++)
			tGraphs.add(sampledGraphs.get(i).adaptiveThresholding(t));
		List<int[][]> wGraphs = new ArrayList<int[][]>();
		for (int i = 0; i < this.size; i++)
			wGraphs.add(wg.sampledGraphs.get(i).adaptiveThresholding(t));

		return this.submodularSubmodularSearch(tGraphs, wGraphs, overlap);
	}

	public static boolean graphCompare(int[][] graph1, int[][] graph2) {
		for (int i = 0; i < graph1.length; i++) {
			for (int j = 0; j < graph2.length; j++) {
				if (graph1[i][j] != graph2[i][j])
					return false;
			}
		}
		return true;
	}

	public int[][] submodularSearch(List<int[][]> graphs, int e) {
		int[][] sGraph = new int[this.wishartMeanPrecision.length][this.wishartMeanPrecision.length];
		int selected = 0;
		while (selected < e) {
			int i_index = -1, j_index = -1, maxV = -1;
			for (int i = 0; i < this.wishartMeanPrecision.length; i++) {
				for (int j = 0; j < this.wishartMeanPrecision.length; j++) {
					if (sGraph[i][j] == 0) {
						int v = 0;
						for (int[][] graph : graphs) {
							if (graph[i][j] != 0)
								v++;
						}
						if (v > maxV) {
							maxV = v;
							i_index = i;
							j_index = j;
						}
					}
				}
			}
			if (maxV <= 0)
				return sGraph;
			sGraph[i_index][j_index] = 1;
			selected++;
		}
		return sGraph;
	}

	public int[][] submodularSubmodularSearch(List<int[][]> graphs, List<int[][]> wgraphs, double overlap) {
		int[][] sGraph = new int[this.wishartMeanPrecision.length][this.wishartMeanPrecision.length];
		double error = 0;
		while (error < overlap * size) {
			int i_index = -1, j_index = -1;
			double maxV = -1;
			double ecost = 0;
			for (int i = 0; i < this.wishartMeanPrecision.length; i++) {
				for (int j = 0; j < this.wishartMeanPrecision.length; j++) {
					if (sGraph[i][j] == 0) {
						double v = 0;
						for (int[][] graph : graphs) {
							if (graph[i][j] != 0)
								v++;
						}
						double u = 0;
						for (int[][] graph : wgraphs) {
							if (graph[i][j] != 0)
								u++;
						}

						if (v / u > maxV) {
							maxV = v / u;
							i_index = i;
							j_index = j;
							ecost = u;
						}
					}
				}
			}
			if (maxV <= 0)
				return sGraph;
			sGraph[i_index][j_index] = 1;
			error += ecost;
		}
		return sGraph;
	}

	public static Matrix sample(final Random random, final Vector mean, final Matrix covarianceSqrt,
			final int degreesOfFreedom) {
		ArrayList<Vector> xs = MultivariateGaussian.sample(mean, covarianceSqrt, random, degreesOfFreedom);
		RingAccumulator<Matrix> sum = new RingAccumulator<Matrix>();
		for (Vector x : xs) {
			sum.accumulate(x.outerProduct(x));
		}
		return sum.getSum();
	}

}
