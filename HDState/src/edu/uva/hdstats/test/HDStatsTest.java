package edu.uva.hdstats.test;

import edu.uva.hdstats.*;
import edu.uva.libopt.numeric.Utils;

public class HDStatsTest {
	
	public static void main(String[] args){
		double[][] samples=Utils.getSparseRandomMatrix(60, 30,0.1);
		System.out.println("************Samples Generated*************");
	//	Estimator est=new PDLassoEstimator(0.01);
		double[][] hdcovar=new LDEstimator().covariance(samples);
		double[][] ihdcovar=new DiagKeptSparseCovEstimator(0.2,5).covariance(samples);
		double[][] ldcovar=new LDEstimator().covariance(samples);
		double[][] glcovar=new NonSparseEstimator(0.2).covariance(samples);

		double error1=0;
		double error2=0;
		double error3=0;
		double error4=0;

		double basis=0;
		
		System.out.println("sample estimation");

		for(int i=0;i<hdcovar.length;i++){
			for(int j=0;j<hdcovar[i].length;j++){
				System.out.print(hdcovar[i][j]+"\t");
			//	basis+=Math.abs(ldcovar[i][j]);
			}
			System.out.println();
		}
		
		System.out.println("glasso estimation");

		
		for(int i=0;i<hdcovar.length;i++){
			for(int j=0;j<hdcovar[i].length;j++){
				error1+=Math.abs(hdcovar[i][j]-ldcovar[i][j]);
//				error2+=Math.abs(ihdcovar[i][j]-ldcovar[i][j]);
				error3+=Math.abs(glcovar[i][j]-ldcovar[i][j]);
//				error4+=Math.abs(ihdcovar[i][j]-glcovar[i][j]);
				System.out.print(glcovar[i][j]+"\t");
				basis+=Math.abs(ldcovar[i][j]);
			}
			System.out.println();
		}
		System.out.println("daehr estimation");

		for(int i=0;i<hdcovar.length;i++){
			for(int j=0;j<hdcovar[i].length;j++){
				System.out.print(ihdcovar[i][j]+"\t");
				basis+=Math.abs(ldcovar[i][j]);
			}
			System.out.println();
		}
		
		System.out.println("error:\t"+(error1/basis)+"\t"+(error2/basis)+"\t"+(error3/basis)+"\t"+(error4/basis));

	}

}
