package xiong.hdstats.graph;

import xiong.hdstats.MLEstimator;
import xiong.hdstats.distribution.MultivariateGaussianDistribution;
import xiong.hdstats.distribution.MultivariateGaussianMixture;

public class GMMEstimator extends MLEstimator{
	public MultivariateGaussianMixture gmm;
	
	public GMMEstimator(double[][] data, int k){
		gmm=new MultivariateGaussianMixture(data,k);
	}
}
