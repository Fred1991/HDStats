package xiong.hdstats.graph;

import smile.stat.distribution.MultivariateGaussianDistribution;
import smile.stat.distribution.MultivariateGaussianMixture;
import xiong.hdstats.MLEstimator;

public class GMMEstimator extends MLEstimator{
	public MultivariateGaussianMixture gmm;
	
	public GMMEstimator(double[][] data, int k){
		gmm=new MultivariateGaussianMixture(data,k);
	}
}
