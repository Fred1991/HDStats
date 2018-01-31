package xiong.hdstats.da.comb;

import Jama.Matrix;
import xiong.hdstats.da.BetaLDA;
import xiong.hdstats.da.PseudoInverseLDA;
import xiong.hdstats.da.shruken.InvalidLDA;
import xiong.hdstats.gaussian.CovarianceEstimator;
import xiong.hdstats.gaussian.GLassoEstimator;
import xiong.hdstats.gaussian.SampleCovarianceEstimator;
import xiong.hdstats.gaussian.SpikedUpperEstimator;
import xiong.hdstats.opt.GradientDescent;
import xiong.hdstats.opt.comb.OrthogonalMatchingPursuit;
import xiong.hdstats.opt.comb.TruncatedRayleighFlow;
import xiong.hdstats.opt.var.MatrixMVariable;

public class OMPDA extends BetaLDA {
	private OrthogonalMatchingPursuit omp;
	private int k;

	public OMPDA(double[][] d, int[] g, boolean p, int _k) {
		this.k = _k;
		this.init(d, new double[d[0].length][d[0].length], g);
		double[][] _g = new double[g.length][1];
		for (int i = 0; i < g.length; i++) {
			if (g[i] > 0)
				_g[i][0] = 1.0;
			else
				_g[i][0] = -1.0;
		}
		omp = new OrthogonalMatchingPursuit(new Matrix(d), new Matrix(_g), this.k);
		double[][] _init_arr =new double[1][d[0].length];
		double[] _beta = this.getBeta();
		for(int i=0;i<_beta.length;i++)
			_init_arr[0][i] = _beta[i];
		Matrix _init = new Matrix(_init_arr).transpose();
		MatrixMVariable result = (MatrixMVariable) GradientDescent.getMinimum(omp, new MatrixMVariable(_init), 1e-4,
				1e-1, 100, GradientDescent.GD);
		this.beta[2] = result.getMtx();
	}

}
