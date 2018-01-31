package xiong.hdstats.da.shruken;

import Jama.Matrix;
import xiong.hdstats.da.BetaLDA;
import xiong.hdstats.da.PseudoInverseLDA;
import xiong.hdstats.gaussian.GLassoEstimator;
import xiong.hdstats.gaussian.SpikedUpperEstimator;
import xiong.hdstats.opt.comb.TruncatedRayleighFlow;

public class DBSpikedDA extends BetaLDA {
	private TruncatedRayleighFlow TRF;
	private int k;

	public DBSpikedDA(double[][] d, int[] g, boolean p, int k) {
		this.k = k;
		InvalidLDA ilda = new InvalidLDA(d, g, p);
		SpikedUpperEstimator se = new SpikedUpperEstimator(k);
		Matrix sgraph = new Matrix(se.inverseCovariance(ilda.recenterData));
		Matrix sampCov = new Matrix(ilda.getSampleCovarianceMatrix());
		double[][] graph = sgraph.times(2).minus(sgraph.times(sampCov.times(sgraph))).getArrayCopy();
		this.init(d, graph, g);
	}

	public void iterate(int n) {
		for (int i = 0; i < n; i++)
			TRF.iterate();
		this.beta[2] = TRF.getVector();
	}

}
