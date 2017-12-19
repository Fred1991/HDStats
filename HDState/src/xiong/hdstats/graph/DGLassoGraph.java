package xiong.hdstats.graph;

import xiong.hdstats.gaussian.CovarianceEstimator;
import xiong.hdstats.gaussian.DBGLassoEstimator;
import xiong.hdstats.gaussian.SampleCovarianceEstimator;

public class DGLassoGraph extends SampleCovarianceEstimator{

	public double[][] gaussianPrecision;
	private DBGLassoEstimator ne=new DBGLassoEstimator();

	public DGLassoGraph(double[][] covar,double lambda, boolean s){
		CovarianceEstimator.lambda=lambda;
		this.gaussianPrecision=ne._deSparsifiedGlassoPrecisionMatrix(covar);
	}
	
	public DGLassoGraph(double[][] data,double lambda){
		CovarianceEstimator.lambda=lambda;
		this.covariance(data);
	}
	
	@Override
	public double[][] covariance(double[][] samples) {
		this.gaussianPrecision=ne._deSparsifiedGlassoPrecisionMatrix(super.covariance(samples));
		return inverse(this.gaussianPrecision);
	}
	
	public int[][] thresholding(double threshold){
		return SampleGraph._thresholding(threshold,gaussianPrecision);
	}
	
	public int[][] adaptiveThresholding(double threshold){
		return SampleGraph._adaptiveThresholding(threshold,gaussianPrecision);
	}
	
	public int[][] thresholdingDiff(double t, DGLassoGraph wg) {
		int[][] graph=this.thresholding(t);
		int[][] wgraph=wg.thresholding(t);
		int[][] dgraph=new int[graph.length][graph.length];
		for(int i=0;i<dgraph.length;i++){
			for(int j=0;j<dgraph.length;j++){
				if(graph[i][j]!=wgraph[i][j])
					dgraph[i][j]=1;
			}
		}
		return dgraph;
	}
	
	public int[][] adaptiveThresholdingDiff(double t, DGLassoGraph wg) {
		int[][] graph=this.adaptiveThresholding(t);
		int[][] wgraph=wg.adaptiveThresholding(t);
		int[][] dgraph=new int[graph.length][graph.length];
		for(int i=0;i<dgraph.length;i++){
			for(int j=0;j<dgraph.length;j++){
				if(graph[i][j]!=wgraph[i][j])
					dgraph[i][j]=1;
			}
		}
		return dgraph;
	}
}
