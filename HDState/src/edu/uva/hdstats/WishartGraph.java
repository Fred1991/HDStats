package edu.uva.hdstats;

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

public class WishartGraph extends MLEstimator {

	public double[][] wishartMeanPrecision;
	public double[] mean;
	public List<SampleGraph> sampledGraphs = new ArrayList<SampleGraph>();
	public int size;
	private NonSparseEstimator ne = new NonSparseEstimator();

	public WishartGraph(double[][] data, double lambda, int size, double K) {
		Estimator.lambda = lambda;
		this.size = size;
		this.covariance(data);
		this.mean = this.getMean(data);

		for (int i = 0; i < this.wishartMeanPrecision.length; i++)
			this.wishartMeanPrecision[i][i] += K;

		VectorFactory vf = VectorFactory.getDenseDefault();
		MatrixFactory mf = MatrixFactory.getDenseDefault();
		Vector meanV = vf.copyArray(mean);
		Matrix covarianceSqrt = CholeskyDecompositionMTJ
				.create(DenseMatrixFactoryMTJ.INSTANCE.copyMatrix(mf.copyArray(wishartMeanPrecision).inverse())).getR();
		int fDOF = Math.min(Math.max(this.wishartMeanPrecision.length, data.length) * 10, 2000);
		Random r = new Random(System.currentTimeMillis());
		while (sampledGraphs.size() < this.size) {
			Matrix mtx = sample(r, meanV, covarianceSqrt, fDOF);
			sampledGraphs.add(new SampleGraph(mtx.inverse().toArray(), false));
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

	
	public int[][] thresholdingDiff(double t, double e, WishartGraph wg) {
		List<int[][]> tGraphs = new ArrayList<int[][]>();
		for (int i = 0; i < this.size; i++)
			tGraphs.add(sampledGraphs.get(i).thresholding(t));
		List<int[][]> wGraphs = new ArrayList<int[][]>();
		for (int i = 0; i < this.size; i++)
			wGraphs.add(wg.sampledGraphs.get(i).thresholding(t));
		
		return this.submodularSubmodularSearch(tGraphs, wGraphs, e);
	}

	public int[][] adaptiveThresholdingDiff(double t, double e, WishartGraph wg) {
		List<int[][]> tGraphs = new ArrayList<int[][]>();
		for (int i = 0; i < this.size; i++)
			tGraphs.add(sampledGraphs.get(i).adaptiveThresholding(t));
		List<int[][]> wGraphs = new ArrayList<int[][]>();
		for (int i = 0; i < this.size; i++)
			wGraphs.add(wg.sampledGraphs.get(i).adaptiveThresholding(t));

		return this.submodularSubmodularSearch(tGraphs, wGraphs, e);
	}

	
	
	public int[][] submodularSearch(List<int[][]> graphs, int e) {
		int[][] sGraph = new int[this.wishartMeanPrecision.length][this.wishartMeanPrecision.length];
		int selected = 0;
		while (selected < e) {
			int i_index=-1, j_index=-1, maxV=-1;
			for (int i = 0; i < this.wishartMeanPrecision.length; i++) {
				for (int j = 0; j < this.wishartMeanPrecision.length; j++) {
					if (sGraph[i][j] == 0) {
						int v=0;
						for(int[][] graph:graphs){
							if(graph[i][j]!=0)
								v++;
						}
						if(v>maxV){
							maxV=v;
							i_index=i;
							j_index=j;
						}
					}
				}
			}
			sGraph[i_index][j_index]=1;
			selected++;
		}
		return sGraph;
	}
	
	public int[][] submodularSubmodularSearch(List<int[][]> graphs, List<int[][]> wgraphs, double e) {
		int[][] sGraph = new int[this.wishartMeanPrecision.length][this.wishartMeanPrecision.length];
		double error = 0;
		while (error < e) {
			int i_index=-1, j_index=-1; 
			double maxV=-1;
			double ecost=0;
			for (int i = 0; i < this.wishartMeanPrecision.length; i++) {
				for (int j = 0; j < this.wishartMeanPrecision.length; j++) {
					if (sGraph[i][j] == 0) {
						double v=0;
						for(int[][] graph:graphs){
							if(graph[i][j]!=0)
								v++;
						}
						double u=0;
						for(int[][] graph:wgraphs){
							if(graph[i][j]!=0)
								u++;
						}
						
						if(v/u>maxV){
							maxV=v/u;
							i_index=i;
							j_index=j;
							ecost=u;
						}
					}
				}
			}
			sGraph[i_index][j_index]=1;
			error+=ecost;
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
