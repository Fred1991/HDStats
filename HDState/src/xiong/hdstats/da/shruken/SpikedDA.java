package xiong.hdstats.da.shruken;

import Jama.Matrix;
import xiong.hdstats.da.BetaLDA;
import xiong.hdstats.da.PseudoInverseLDA;
import xiong.hdstats.gaussian.GLassoEstimator;
import xiong.hdstats.gaussian.SpikedLowerEstimator;
import xiong.hdstats.gaussian.SpikedUpperEstimator;
import xiong.hdstats.opt.comb.TruncatedRayleighFlow;

public class SpikedDA extends BetaLDA {
	private TruncatedRayleighFlow TRF;
	private int k;

	public SpikedDA(double[][] d, int[] g, boolean p, int k) {
		this.k = k;
		InvalidLDA ilda = new InvalidLDA(d, g, p);
		SpikedLowerEstimator se = new SpikedLowerEstimator(k);
		double[][] graph = se.inverseCovariance(ilda.recenterData);
		this.init(d, graph, g);
	}

	public void iterate(int n) {
		for (int i = 0; i < n; i++)
			TRF.iterate();
		this.beta[2] = TRF.getVector();
	}

}
