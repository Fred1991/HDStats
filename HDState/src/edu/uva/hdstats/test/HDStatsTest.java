package edu.uva.hdstats.test;

import edu.uva.hdstats.Estimator;
import edu.uva.hdstats.LassoEstimator;
import edu.uva.libopt.numeric.Utils;

public class HDStatsTest {
	
	public static void main(String[] args){
		double[][] samples=Utils.getSparseRandomMatrix(4000, 200,1);
		System.out.println("************Samples Generated*************");
		Estimator est=new LassoEstimator(0.01);
		System.out.println(est.covariance(samples).length);

	}

}
