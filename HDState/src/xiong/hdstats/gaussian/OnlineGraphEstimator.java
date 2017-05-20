package xiong.hdstats.gaussian;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import Jama.Matrix;
import xiong.hdstats.MLEstimator;

public class OnlineGraphEstimator extends MLEstimator {
	public double[][] graph;
	public double[][] init_cov;
	int k;


	public OnlineGraphEstimator(int k) {
		this.k=k;
	}	
	
	public OnlineGraphEstimator() {
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
	

	
	@Override
	public double[][] covariance(double[][] samples) {
		// TODO Auto-generated method stub
		double[][] covar=new double[samples[0].length][samples[0].length];
		for(int i=0;i<samples.length;i++){
			for(int j=0;j<samples[i].length;j++){
				for(int k=0;k<samples[i].length;k++){
					covar[j][k]+=(samples[i][j])*(samples[i][k])/samples.length;
				}
			}
		}
		System.out.println("************LD Covariance Estimated*************");
		return covar;
	}

	public void init(double[][] samples){
		this.init_cov=this.covariance(samples);
		this.graph=new Matrix(this.init_cov).inverse().getArray();
		//HT(k,this.graph);
	}

}
