package xiong.hdstats.da.shruken;


import xiong.hdstats.da.BetaLDA;
import xiong.hdstats.da.PseudoInverseLDA;
import xiong.hdstats.gaussian.CovarianceEstimator;
import xiong.hdstats.gaussian.GLassoEstimator;

public class SDABeta extends BetaLDA {
//	private TruncatedRayleighFlow TRF;

	public SDABeta(double[][] d, int[] g, boolean p) {
		PseudoInverseLDA olda=new PseudoInverseLDA(d,g,p);
		double[][] AMat = olda.pooledCovariance;
		GLassoEstimator nse =new GLassoEstimator(CovarianceEstimator.lambda);
		double[][] graph = nse._glassoPrecisionMatrix(AMat);
		this.init(d, graph, g);
	//	TRF = new TruncatedRayleighFlow(this.k, 1.0e-4, AMat, BMat);
	//	TRF.init(this.beta[2].transpose().getArrayCopy()[0]);
	//	this.iterate(1000);
	}
	

}
