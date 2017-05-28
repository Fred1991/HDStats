package xiong.hdstats.da;


import Jama.Matrix;
import xiong.hdstats.Estimator;
import xiong.hdstats.gaussian.NonSparseEstimator;
import xiong.hdstats.graph.PDLassoEstimator;

public class L0NormLDAClassifier extends BetaLDA {
	private TruncatedRayleighFlow TRF;
	private int k;

	public L0NormLDAClassifier(double[][] d, int[] g, boolean p, int k) {
		this.k = k;
		OLDA olda=new OLDA(d,g,p);
		double[][] AMat = olda.pooledCovariance;
		Estimator.lambda = 10.0;
		NonSparseEstimator nse =new NonSparseEstimator();
		double[][] graph = nse._deSparsifiedGlassoPrecisionMatrix(AMat);
		this.init(d, graph, g);
		double[][] BMat = new double[d[0].length][d[0].length];
		for(int i=0;i<2;i++){
			for(int j=0;j<d[0].length;j++){
				for(int l=0;l<d[0].length;l++){
					BMat[j][l]+=((double)frequencies[i]/(double)totalNum)*means[i][0][j]*means[i][0][l];
				}
			}
		}
		
		TRF = new TruncatedRayleighFlow(this.k, 1.0e-6, AMat, BMat);
		TRF.init(this.beta[2].transpose().getArrayCopy()[0]);
		this.iterate(1000);
	}

	public void iterate(int n){
		for(int i=0;i<n;i++)
			TRF.iterate();
		this.beta[2]=TRF.getVector();
	}
	
	


}
