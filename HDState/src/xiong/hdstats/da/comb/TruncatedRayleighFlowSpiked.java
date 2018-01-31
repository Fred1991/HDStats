package xiong.hdstats.da.comb;

import Jama.Matrix;
import xiong.hdstats.da.BetaLDA;
import xiong.hdstats.da.PseudoInverseLDA;
import xiong.hdstats.da.shruken.InvalidLDA;
import xiong.hdstats.gaussian.CovarianceEstimator;
import xiong.hdstats.gaussian.DBGLassoEstimator;
import xiong.hdstats.gaussian.SampleCovarianceEstimator;
import xiong.hdstats.gaussian.SpikedLowerEstimator;
import xiong.hdstats.gaussian.SpikedUpperEstimator;
import xiong.hdstats.graph.PDLassoEstimator;
import xiong.hdstats.opt.comb.TruncatedRayleighFlow;

public class TruncatedRayleighFlowSpiked extends BetaLDA {
	private TruncatedRayleighFlow TRF;
	private int k;

	public TruncatedRayleighFlowSpiked(double[][] d, int[] g, boolean p, int _k, int _nz) {
		this.k = _nz;
		InvalidLDA ilda = new InvalidLDA(d,g,p);
		SpikedUpperEstimator se =new SpikedUpperEstimator(_k);
		double[][] graph = //new double[d[0].length][d[0].length];// 
					se.inverseCovariance(ilda.recenterData);
		this.init(d, graph, g);
		double[][] AMat = se.covariance(ilda.recenterData);
		double[][] BMat = new double[d[0].length][d[0].length];
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < d[0].length; j++) {
				for (int l = 0; l < d[0].length; l++) {
					BMat[j][l] += ((double) frequencies[i] / (double) totalNum) * means[i][0][j] * means[i][0][l];
				}
			}
		}
		
		//OMPDA init= new OMPDA(d,g,p,_nz);
		TRF = new TruncatedRayleighFlow(this.k, 1.0e-4, 1.0e-1, AMat, BMat);
		TRF.init(this.getBeta());
		this.iterate(100);
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
