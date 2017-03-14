package edu.uva.hdstats.test;

import edu.uva.hdstats.*;
import edu.uva.hdstats.gaussian.GLassoEstimator;
import edu.uva.hdstats.graph.NonSparseEstimator;
import edu.uva.libopt.numeric.Utils;

public class HDStatsTest {
	
	public static void main(String[] args){
		double[][] samples=Utils.getSparseRandomMatrix(200, 300,0.1);
		System.out.println("************Samples Generated*************");
	//	Estimator est=new PDLassoEstimator(0.01);
		double[][] spl_cov=new MLEstimator().covariance(samples);
		double[][] l1p_cov=new DiagKeptSparseCovEstimator(0.001,5).covariance(samples);
		double[][] gls_cov=new GLassoEstimator(0.001).covariance(samples);
		double[][] ngl_cov=new NonSparseEstimator(0.001).covariance(samples);

		double error1=0;
		double error2=0;
		double error3=0;
		double error4=0;

		double basis=0;
		
		System.out.println();		
		System.out.println("sample estimation");

		for(int i=0;i<spl_cov.length;i++){
			for(int j=0;j<spl_cov[i].length;j++){
				System.out.print(spl_cov[i][j]+"\t");
			//	basis+=Math.abs(ldcovar[i][j]);
			}
			System.out.println();
		}
		
		System.out.println();
		System.out.println("glasso estimation");

		
		for(int i=0;i<spl_cov.length;i++){
			for(int j=0;j<spl_cov[i].length;j++){
				error1+=Math.abs(spl_cov[i][j]-l1p_cov[i][j]);
				error2+=Math.abs(spl_cov[i][j]-gls_cov[i][j]);
				error3+=Math.abs(spl_cov[i][j]-ngl_cov[i][j]);
				error4+=Math.abs(ngl_cov[i][j]-gls_cov[i][j]);
				System.out.print(gls_cov[i][j]+"\t");
				basis=1;
			}
			System.out.println();
		}
		
		System.out.println();
		System.out.println("de-sparsified glasso estimation");

		for(int i=0;i<spl_cov.length;i++){
			for(int j=0;j<spl_cov[i].length;j++){
				System.out.print(ngl_cov[i][j]+"\t");
			//	basis+=Math.abs(ldcovar[i][j]);
			}
			System.out.println();
		}
		
		
		System.out.println();
		System.out.println("daehr estimation");

		for(int i=0;i<spl_cov.length;i++){
			for(int j=0;j<spl_cov[i].length;j++){
				System.out.print(l1p_cov[i][j]+"\t");
			//	basis+=Math.abs(gls_cov[i][j]);
			}
			System.out.println();
		}
		
		System.out.println("error:\t"+(error1/basis)+"\t"+(error2/basis)+"\t"+(error3/basis)+"\t"+(error4/basis));

	}

}
