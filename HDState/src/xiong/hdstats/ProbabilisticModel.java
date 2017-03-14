package xiong.hdstats;

public interface ProbabilisticModel {
	
	public void buildModel(double[][] covariance, double[] mean);
	public double getProbability(double[] vec);

}
