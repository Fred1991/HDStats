package xiong.hdstats.da.comb;

import Jama.Matrix;
import xiong.hdstats.Estimator;
import xiong.hdstats.da.BetaLDA;
import xiong.hdstats.da.PseudoInverseLDA;
import xiong.hdstats.gaussian.DBGLassoEstimator;
import xiong.hdstats.graph.PDLassoEstimator;
import xiong.hdstats.opt.comb.TruncatedRayleighFlow;

public class TruncatedRayleighFlowDBSDA extends BetaLDA {
	private TruncatedRayleighFlow TRF;
	private int k;

	public TruncatedRayleighFlowDBSDA(double[][] d, int[] g, boolean p, int _k) {
		this.k = _k;
		PseudoInverseLDA olda = new PseudoInverseLDA(d, g, p);
		double[][] AMat = olda.pooledCovariance;
		DBGLassoEstimator nse = new DBGLassoEstimator(Estimator.lambda);
		double[][] graph = nse._deSparsifiedGlassoPrecisionMatrix(AMat);
		this.init(d, graph, g);
		// if(d.length<d[0].length*2)
		AMat = new Matrix(graph).inverse().getArrayCopy();
		double[][] BMat = new double[d[0].length][d[0].length];
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < d[0].length; j++) {
				for (int l = 0; l < d[0].length; l++) {
					BMat[j][l] += ((double) frequencies[i] / (double) totalNum) * means[i][0][j] * means[i][0][l];
				}
			}
		}

		TRF = new TruncatedRayleighFlow(this.k, 1.0e-4, 1.0e-3, AMat, BMat);
		TRF.init(this.beta[2].transpose().getArrayCopy()[0]);
		this.iterate(1000);
	}

	public void iterate(int n) {
		boolean cont = true;
		int count =0;
		while(cont ==true &&count<n){
			cont = TRF.iterate();
			count ++;
		}
		
		this.beta[2] = TRF.getVector();
	}

}
