package xiong.hdstats.da.comb;

import smile.stat.distribution.SpikedMultivariateGaussianDistribution;
import smile.stat.distribution.MultivariateGaussianDistribution;
import xiong.hdstats.da.BetaLDA;
import xiong.hdstats.da.PseudoInverseLDA;
import xiong.hdstats.gaussian.CovarianceEstimator;
import xiong.hdstats.gaussian.DBGLassoEstimator;
import xiong.hdstats.opt.comb.RayleighFlow;

public class RayleighFlowLDA extends BetaLDA {
	private RayleighFlow TRF;

	public RayleighFlowLDA(double[][] d, int[] g, boolean p) {
		PseudoInverseLDA olda = new PseudoInverseLDA(d, g, p);
		double[][] AMat = olda.pooledCovariance;
		this.init(d, olda.pooledInverseCovariance, g);

		double[][] BMat = new double[d[0].length][d[0].length];
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < d[0].length; j++) {
				for (int l = 0; l < d[0].length; l++) {
					BMat[j][l] += ((double) frequencies[i] / (double) totalNum) * means[i][0][j] * means[i][0][l];
				}
			}
		}
		
		TRF = new RayleighFlow(1.0e-4, AMat, BMat);
//		double[] ibeta=new double[d[0].length];
//		for(int i=0;i<d[0].length;i++){
//			ibeta[i]=1;
//		} 
		TRF.init(this.beta[2].transpose().getArrayCopy()[0]);
		this.iterate(200);
	}

	public void iterate(int n) {
		for (int i = 0; i < n; i++)
			TRF.iterate();
		this.beta[2] = TRF.getVector();
	}

}
