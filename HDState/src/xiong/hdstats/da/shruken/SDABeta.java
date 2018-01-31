package xiong.hdstats.da.shruken;


import xiong.hdstats.da.BetaLDA;
import xiong.hdstats.da.PseudoInverseLDA;
import xiong.hdstats.gaussian.CovarianceEstimator;
import xiong.hdstats.gaussian.GLassoEstimator;

public class SDABeta extends BetaLDA {
//	private TruncatedRayleighFlow TRF;

	public SDABeta(double[][] d, int[] g, boolean p) {
		InvalidLDA olda=new InvalidLDA(d,g,p);
		double[][] AMat = olda.getSampleCovarianceMatrix();
		GLassoEstimator nse =new GLassoEstimator(CovarianceEstimator.lambda);
		double[][] graph = nse._glassoPrecisionMatrix(AMat);
		this.init(d, graph, g);
	}
	

}
