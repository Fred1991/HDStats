package xiong.hdstats.da.shruken;


import xiong.hdstats.da.BetaLDA;
import xiong.hdstats.da.PseudoInverseLDA;
import xiong.hdstats.gaussian.CovarianceEstimator;
import xiong.hdstats.gaussian.DBGLassoEstimator;

public class DBSDABeta extends BetaLDA {
//	private TruncatedRayleighFlow TRF;

	public DBSDABeta(double[][] d, int[] g, boolean p) {
		PseudoInverseLDA olda=new PseudoInverseLDA(d,g,p);
		double[][] AMat = olda.pooledCovariance;
		DBGLassoEstimator nse =new DBGLassoEstimator(CovarianceEstimator.lambda);
		double[][] graph = nse._deSparsifiedGlassoPrecisionMatrix(AMat);
		this.init(d, graph, g);
	//	TRF = new TruncatedRayleighFlow(this.k, 1.0e-4, AMat, BMat);
	//	TRF.init(this.beta[2].transpose().getArrayCopy()[0]);
	//	this.iterate(1000);
	}
	

}
