package xiong.hdstats.da;

import Jama.Matrix;
import xiong.hdstats.Estimator;
import xiong.hdstats.gaussian.NonSparseEstimator;

public class StochasticTruncatedRayleighFlowDBSDA extends BetaLDA {
	private StochasticTruncatedRayleighFlow TRF;

	public StochasticTruncatedRayleighFlowDBSDA(double[][] d, int[] g, boolean p, int k, double noise) {
		OLDA olda = new OLDA(d, g, p);
		double[][] AMat = olda.pooledCovariance;
		NonSparseEstimator nse = new NonSparseEstimator(Estimator.lambda);
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

		TRF = new StochasticTruncatedRayleighFlow(1.0e-4, AMat, BMat, k, noise);
		TRF.init(this.beta[2].transpose().getArrayCopy()[0]);
		this.iterate(200);
	}

	public void iterate(int n) {
		for (int i = 0; i < n; i++)
			TRF.iterate();
		this.beta[2] = TRF.getVector();
	}

}
