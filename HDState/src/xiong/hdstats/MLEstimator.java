package xiong.hdstats;

public class MLEstimator extends Estimator{
	final public static String R_src_Path="/Users/xiongha/R_src/";
	
	@Override
	public double[][] covariance(double[][] samples) {
		// TODO Auto-generated method stub
		double[][] covar=new double[samples[0].length][samples[0].length];
		double[] mean=this.getMean(samples);
		for(int i=0;i<samples.length;i++){
			for(int j=0;j<samples[i].length;j++){
				for(int k=0;k<samples[i].length;k++){
					covar[j][k]+=(samples[i][j]-mean[j])*(samples[i][k]-mean[k])/samples.length;
				}
			}
		}
		System.out.println("************LD Covariance Estimated*************");
		return covar;
	}

	@Override
	public double[] getMean(double[][] samples) {
		// TODO Auto-generated method stub
		double[] mean=new double[samples[0].length];
		for(int i=0;i<samples.length;i++){
			for(int j=0;j<samples[i].length;j++){
				mean[j]+=samples[i][j]/mean.length;
			}
		}

		return mean;
	}

	@Override
	public void covarianceApprox(double[][] samples) {
		// TODO Auto-generated method stub
		//return samples;
	}

}
