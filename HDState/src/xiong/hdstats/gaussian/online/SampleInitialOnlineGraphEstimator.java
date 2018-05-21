package xiong.hdstats.gaussian.online;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import Jama.Matrix;
import xiong.hdstats.gaussian.SampleCovarianceEstimator;

public class SampleInitialOnlineGraphEstimator implements OnlineGraphEstimator {
	public double[][] graph;
	Matrix _innerCore;
	public double[][] init_cov;
	public double _lambda;

	public SampleInitialOnlineGraphEstimator(double lambda) {
		this._lambda = lambda;
	}

	public void update(int index, double[] newdata) {
		// double[][] add=new double[newdata.length][newdata.length];
		// for(int i=0;i<newdata.length;i++){
		// for(int j=0;j<newdata.length;j++){
		// add[i][j]=newdata[i]*newdata[j];
		// }
		// }
		Matrix nDMtx = new Matrix(newdata, newdata.length);
		// Matrix Ai=new Matrix(this.graph).times(index/(index-1.0));
		Matrix Ai = this._innerCore;
		// Matrix B = nDMtx.times(nDMtx.transpose()).times(1.0/index);
		Matrix B = nDMtx.times(nDMtx.transpose());
		// Matrix B=new Matrix(add).times(1.0/index);
		Matrix additive = ((Ai.times(B)).times(Ai)).times(-1.0 / (1 + (B.times(Ai)).trace()));
		// Matrix additive=((Ai.times(B)).times(Ai)).times(-1.0);
		// graph=(Ai.plus(additive)).getArray();
		this._innerCore = (Ai.plus(additive));
		graph = this._innerCore.times(index).getArray();
		// HT(k,this.graph);
	}

	public int getL0Norm() {
		int l0norm = 0;
		for (int i = 0; i < graph.length; i++) {
			for (int j = 0; j < graph.length; j++) {
				if (graph[i][j] != 0)
					l0norm++;
			}
		}
		return l0norm;
	}

	public void init(double[][] samples) {
		this.init_cov = new SampleCovarianceEstimator().covariance(samples);
		Matrix D = new Matrix(this.init_cov).eig().getD();
		List<Double> diags = new ArrayList<Double>();
		for (int i = 0; i < D.getColumnDimension(); i++)
			diags.add(D.get(i, i));
		Collections.sort(diags);
		double _lamb = this._lambda;
		if (diags.get(0) < 0)
			_lamb += -1 * diags.get(0);
		else if (diags.get(0) > _lambda)
			_lamb = 0;

		Matrix lambI = Matrix.identity(init_cov.length, init_cov.length).times(_lamb);
		this._innerCore = (new Matrix(this.init_cov).plus(lambI)).inverse();
		this.graph = this._innerCore.getArray();
		this._innerCore.times(1.0 / samples.length);
		// HT(k,this.graph);
	}

	@Override
	public double[][] getGraph() {
		// TODO Auto-generated method stub
		return this.graph;
	}

}
