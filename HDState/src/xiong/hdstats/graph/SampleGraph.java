package xiong.hdstats.graph;

import xiong.hdstats.MLEstimator;

public class SampleGraph extends MLEstimator{

	public double[][] gaussianPrecision;
	
	public SampleGraph(double[][] data){
		this.covariance(data);
	}

	
	public SampleGraph(double[][] precision, boolean f){
		this.gaussianPrecision=precision;
	}
	
	
	@Override
	public double[][] covariance(double[][] samples) {
		double[][] sampleEst=super.covariance(samples);
		this.gaussianPrecision=inverse(sampleEst);
		return sampleEst;
	}
	
	public int[][] thresholding(double threshold){
		return _thresholding(threshold,gaussianPrecision);
	}
	
	public int[][] adaptiveThresholding(double threshold){
		return _adaptiveThresholding(threshold,gaussianPrecision);
	}
	
	
	public static int[][] _thresholding(double threshold, double[][] graph){
		int[][] edges=new int[graph.length][graph[0].length];
		for(int i=0;i<graph.length;i++){
			for(int j=0;j<graph[i].length;j++){
				if(Math.abs(graph[i][j])>threshold)
					edges[i][j]=1;
				else
					edges[i][j]=0;
			}
		}
		return edges;
	}
	
	public static int[][] _adaptiveThresholding(double threshold, double[][] graph){
		int[][] edges=new int[graph.length][graph[0].length];
		for(int i=0;i<graph.length;i++){
			for(int j=0;j<graph[i].length;j++){
				if(Math.abs(graph[i][j])>threshold*Math.sqrt(Math.abs(graph[i][i]*graph[j][j])+Math.abs(graph[i][j]*graph[i][j])))
					edges[i][j]=1;
				else
					edges[i][j]=0;
			}
		}
		return edges;
	}
	
	public int[][] thresholdingDiff(double t, SampleGraph wg) {
		int[][] graph=this.thresholding(t);
		int[][] wgraph=wg.thresholding(t);
		int[][] dgraph=new int[graph.length][graph.length];
		for(int i=0;i<dgraph.length;i++){
			for(int j=0;j<dgraph.length;j++){
				if(graph[i][j]!=wgraph[i][j])
					dgraph[i][j]=1;
				else
					dgraph[i][j]=0;
			}
		}
		return dgraph;
	}
	
	public int[][] adaptiveThresholdingDiff(double t, SampleGraph wg) {
		int[][] graph=this.adaptiveThresholding(t);
		int[][] wgraph=wg.adaptiveThresholding(t);
		int[][] dgraph=new int[graph.length][graph.length];
		for(int i=0;i<dgraph.length;i++){
			for(int j=0;j<dgraph.length;j++){
				if(graph[i][j]!=wgraph[i][j])
					dgraph[i][j]=1;
				else
					dgraph[i][j]=0;
			}
		}
		return dgraph;
	}
}