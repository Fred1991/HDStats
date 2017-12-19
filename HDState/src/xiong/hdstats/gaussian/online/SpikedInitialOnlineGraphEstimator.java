package xiong.hdstats.gaussian.online;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import Jama.Matrix;
import xiong.hdstats.gaussian.SampleCovarianceEstimator;
import xiong.hdstats.gaussian.SpikedEstimator;

public class SpikedInitialOnlineGraphEstimator  {
	public double[][] graph;
	public double[][] init_cov;
	public int comp;

	public SpikedInitialOnlineGraphEstimator(int comp) {
		this.comp=comp;
	}
	
	public void update(int index, double[] newdata){
		double[][] add=new double[newdata.length][newdata.length];
		for(int i=0;i<newdata.length;i++){
			for(int j=0;j<newdata.length;j++){
				add[i][j]=newdata[i]*newdata[j];
			}
		}
		Matrix Ai=new Matrix(this.graph).times(index/(index-1.0));
		Matrix B=new Matrix(add).times(1.0/index);
		Matrix additive=((Ai.times(B)).times(Ai)).times(-1.0/(1+(B.times(Ai)).trace()));
	//	Matrix additive=((Ai.times(B)).times(Ai)).times(-1.0);
		graph=(Ai.plus(additive)).getArray();
	//	HT(k,this.graph);
	}
	
	public int getL0Norm(){
		int l0norm=0;
		for(int i=0;i<graph.length;i++){
			for(int j=0;j<graph.length;j++){
				if(graph[i][j]!=0)
					l0norm++;
			}
		}
		return l0norm;
	}
	

	public void init(double[][] samples){
		this.init_cov=new SpikedEstimator(comp).covariance(samples);
		this.graph=new SpikedEstimator(comp).inverseCovariance(samples);
		//HT(k,this.graph);
	}

}
