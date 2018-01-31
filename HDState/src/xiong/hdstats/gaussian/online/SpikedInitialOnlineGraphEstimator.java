package xiong.hdstats.gaussian.online;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import Jama.Matrix;
import xiong.hdstats.gaussian.SampleCovarianceEstimator;
import xiong.hdstats.gaussian.SpikedUpperEstimator;

public class SpikedInitialOnlineGraphEstimator implements OnlineGraphEstimator {
	public Matrix graph;
	public double[][] init_cov;
	public int comp;

	public SpikedInitialOnlineGraphEstimator(int comp) {
		this.comp = comp;
	}

	public void update(int index, double[] newdata) {
		//double[][] add = new double[newdata.length][newdata.length];
		//for (int i = 0; i < newdata.length; i++) {
		//	for (int j = 0; j < newdata.length; j++) {
		//		add[i][j] = newdata[i] * newdata[j] * 1.0 / index;
		//	}
		//}
		Matrix X = new Matrix(newdata,newdata.length);
		Matrix XT= new Matrix(newdata, 1);
		//Matrix Ai = this.graph.times(index / (index - 1.0));
		//Matrix B = new Matrix(add);
		//Matrix additive = ((Ai.times(B)).times(Ai)).times(-1.0 / (1 + (B.times(Ai)).trace()));
		//graph = (Ai.plus(additive));

		 Matrix TH = this.graph.times(index / (index - 1.0));
		 Matrix XXT = X.times(XT).times(1.0/index);
		 Matrix B = XXT.times(TH);
		 graph = TH.minus(TH.times(B).times(1.0 / (1 + B.trace())));
	}

	public int getL0Norm() {
		int l0norm = 0;
		double[][] _graph = this.graph.getArray();
		for (int i = 0; i < _graph.length; i++) {
			for (int j = 0; j < _graph.length; j++) {
				if (_graph[i][j] != 0)
					l0norm++;
			}
		}
		return l0norm;
	}

	public void init(double[][] samples) {
		this.init_cov = new SpikedUpperEstimator(comp).covariance(samples);
		this.graph = new Matrix(new SpikedUpperEstimator(comp).inverseCovariance(samples))
				.plus(Matrix.identity(samples[0].length, samples[0].length).times(1e-4));
		// HT(k,this.graph);
	}

	@Override
	public double[][] getGraph() {
		// TODO Auto-generated method stub
		return this.graph.getArrayCopy();
	}

}
