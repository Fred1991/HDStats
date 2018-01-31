package xiong.hdstats.gaussian;

import xiong.hdstats.mat.RandSVD;

public class SpikedUpperEstimator extends SampleCovarianceEstimator{

	private int _comp;
	
	public SpikedUpperEstimator(int comp){
		this._comp=comp;
	}
	
	@Override
	public double[][] covariance(double[][] samples) {
		// TODO Auto-generated method stub
		return RandSVD.spikedCovarianceMatrix(samples, _comp, _comp, true);
	}
	

	@Override
	public double[][] inverseCovariance(double[][] samples) {
		// TODO Auto-generated method stub
		return RandSVD.spikedInverseCovarianceMatrix(samples, _comp, _comp, true);

	}

}
