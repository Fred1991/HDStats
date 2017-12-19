package xiong.hdstats.graph.ens;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import edu.uva.libopt.csensing.Compressor;
import edu.uva.libopt.csensing.imp.LASSOCompressor;
import edu.uva.libopt.numeric.Utils;
import gov.sandia.cognition.math.RingAccumulator;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.matrix.mtj.DenseMatrixFactoryMTJ;
import gov.sandia.cognition.math.matrix.mtj.decomposition.CholeskyDecompositionMTJ;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import smile.stat.distribution.MultivariateGaussianMixture;
import smile.stat.distribution.MultivariateMixture.Component;
import xiong.hdstats.gaussian.NearPD;
import xiong.hdstats.gaussian.SampleCovarianceEstimator;
import xiong.hdstats.graph.SampleGraph;

public class GMMGraph extends SampleCovarianceEstimator{

	public MultivariateGaussianMixture gmm;
	public List<SampleGraph> sampledGraphs = new ArrayList<SampleGraph>();
	public List<Double> weight=new ArrayList<Double>();
	public int size;
	public int nData;
	public int dimensions;
	//private NonSparseEstimator ne = new NonSparseEstimator();

	public GMMGraph(double[][] data, int size) {
		this.size = size;
		this.nData=data.length;
		this.dimensions=data[0].length;
		this.gmm=new MultivariateGaussianMixture(data,size);
		
		for(Component comp:this.gmm.getComponents()){
			weight.add(comp.priori);
			sampledGraphs.add(new SampleGraph(inverse(comp.distribution.cov()),false));
		}

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

	
	public int[][] thresholdingDiff(double t, double e, GMMGraph wg) {
		List<int[][]> tGraphs = new ArrayList<int[][]>();
		for (int i = 0; i < this.size; i++)
			tGraphs.add(sampledGraphs.get(i).thresholding(t));
		List<int[][]> wGraphs = new ArrayList<int[][]>();
		for (int i = 0; i < this.size; i++)
			wGraphs.add(wg.sampledGraphs.get(i).thresholding(t));
		
		return this.submodularSubmodularSearch(tGraphs, wGraphs, wg.weight, e);
	}

	public int[][] adaptiveThresholdingDiff(double t, double overlap, GMMGraph wg) {
		List<int[][]> tGraphs = new ArrayList<int[][]>();
		for (int i = 0; i < this.size; i++)
			tGraphs.add(sampledGraphs.get(i).adaptiveThresholding(t));
		List<int[][]> wGraphs = new ArrayList<int[][]>();
		for (int i = 0; i < this.size; i++)
			wGraphs.add(wg.sampledGraphs.get(i).adaptiveThresholding(t));

		return this.submodularSubmodularSearch(tGraphs, wGraphs, wg.weight, overlap);
	}

	public static boolean graphCompare(int[][] graph1, int[][] graph2){
		for(int i=0;i<graph1.length;i++){
			for(int j=0;j<graph2.length;j++){
				if(graph1[i][j]!=graph2[i][j])
					return false;
			}
		}
		return true;
	}
	
	
	
	public int[][] submodularSearch(List<int[][]> graphs, int e) {
		int[][] sGraph = new int[dimensions][dimensions];
		int selected = 0;
		while (selected < e) {
			int i_index=-1, j_index=-1;
			double maxV=-1;
			for (int i = 0; i < dimensions; i++) {
				for (int j = 0; j < dimensions; j++) {
					if (sGraph[i][j] == 0) {
						int v=0;
						for(int k=0;k<graphs.size();k++){
							int[][] graph=graphs.get(k);
							if(graph[i][j]!=0)
								v+=this.weight.get(k);
						}
						if(v>maxV){
							maxV=v;
							i_index=i;
							j_index=j;
						}
					}
				}
			}
			if(maxV<=0)
				return sGraph;
			sGraph[i_index][j_index]=1;
			selected++;
		}
		return sGraph;
	}
	
	public int[][] adaptiveThresholdingDiff(double t, double overlap) {
		List<int[][]> tGraphs = new ArrayList<int[][]>();
		for (int i = 0; i < this.size; i++)
			tGraphs.add(sampledGraphs.get(i).adaptiveThresholding(t));

		return this.$_submodularSubmodularSearch(tGraphs, overlap);
	}

	public int[][] $_submodularSubmodularSearch(List<int[][]> graphs, double overlap) {
		int[][] sGraph = new int[this.dimensions][this.dimensions];
		double error = 0;
		while (error < overlap) {
			int i_index = -1, j_index = -1;
			double maxV = -1;
			double ecost = 0;
			for (int i = 0; i < this.dimensions; i++) {
				for (int j = 0; j < this.dimensions; j++) {
					if (sGraph[i][j] == 0) {
						double v = 0;
						double u = 0;

						for(int k=0;k<graphs.size();k++){
							int[][] graph=graphs.get(k);
							if (graph[i][j] != 0)
								v+=weight.get(k);
							else
								u+=weight.get(k);
						}

						if (u!=0&&v / u > maxV) {
							maxV = v / u;
							i_index = i;
							j_index = j;
							ecost = u/graphs.size();
						}else if(u==0&&v!=0){
							maxV = Double.POSITIVE_INFINITY;
							i_index = i;
							j_index = j;
							ecost = u/graphs.size();
						}
					}
				}
			}
			if (maxV <= 0)
				return sGraph;
			sGraph[i_index][j_index] = 1;
			error += ecost;
			System.out.println(ecost);
		}
		return sGraph;
	}

	
	public int[][] submodularSubmodularSearch(List<int[][]> graphs, List<int[][]> wgraphs, List<Double> wweight, double overlap) {
		int[][] sGraph = new int[this.dimensions][this.dimensions];
		double error = 0;
		while (error < overlap*size) {
			int i_index=-1, j_index=-1; 
			double maxV=-1;
			double ecost=0;
			for (int i = 0; i < this.dimensions; i++) {
				for (int j = 0; j < this.dimensions; j++) {
					if (sGraph[i][j] == 0) {
						double v=0;
						for(int k=0;k<graphs.size();k++){
							int[][] graph=graphs.get(k);
							if(graph[i][j]!=0)
								v+=weight.get(k);
						}
						double u=0;
						for(int k=0;k<wgraphs.size();k++){
							int[][] graph=graphs.get(k);							
							if(graph[i][j]!=0)
								u+=wweight.get(k);
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
			if(maxV<=0)
				return sGraph;
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
	
	public static void main(String[] args){
		double[][] data=Utils.getRandomMatrix(232, 2500);
		GMMGraph gmm=new GMMGraph(data,100);
		gmm.adaptiveThresholding(0,10);
	}

}
