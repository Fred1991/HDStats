package xiong.hdstats.da.comb;

import Jama.Matrix;
import xiong.hdstats.Estimator;
import xiong.hdstats.da.BetaLDA;
import xiong.hdstats.da.PseudoInverseLDA;
import xiong.hdstats.gaussian.GLassoEstimator;
import xiong.hdstats.opt.GradientDescent;
import xiong.hdstats.opt.comb.OrthogonalMatchingPursuit;
import xiong.hdstats.opt.comb.TruncatedRayleighFlow;
import xiong.hdstats.opt.var.MatrixMVariable;

public class OMPDA extends BetaLDA {
	private OrthogonalMatchingPursuit omp;
	private int k;

	public OMPDA(double[][] d, int[] g, boolean p, int _k) {
		this.k = _k;
		PseudoInverseLDA olda = new PseudoInverseLDA(d, g, p);
		this.init(d, olda.pooledInverseCovariance, g);
		double[][] _g = new double[g.length][1];
		for (int i = 0; i < g.length; i++) {
			if(g[i]>0)
				_g[i][0] =1.0;
			else
				_g[i][0] = -1.0;
		}

		omp = new OrthogonalMatchingPursuit(new Matrix(d), new Matrix(_g), this.k);
		Matrix _init = new Matrix(d[0].length, 1);
		MatrixMVariable result = (MatrixMVariable) GradientDescent.getMinimum(omp, new MatrixMVariable(_init),
				1e-4, 1e-3, 10000, GradientDescent.GD);
		this.beta[2] = result.getMtx();
	}

}
