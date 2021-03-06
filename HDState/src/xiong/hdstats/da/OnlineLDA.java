package xiong.hdstats.da;

import java.util.HashMap;

import Jama.Matrix;
import xiong.hdstats.gaussian.online.SampleInitialOnlineGraphEstimator;
import xiong.hdstats.gaussian.online.SpikedInitialOnlineGraphEstimator;

public class OnlineLDA implements Classifier<double[]> {
	public SpikedInitialOnlineGraphEstimator oge ;
	public double[][][] means;
	public int[] frequencies = new int[2];
	public double[] c = new double[2];
	public Matrix[] beta = new Matrix[3];
	public double delta = 0;
	public int totalNum;

	public void init(double[][] data, int[] label, int s) {
		this.oge = new SpikedInitialOnlineGraphEstimator(s);
		this.oge.init(data);
		int index = 0;
		means = new double[2][1][data[0].length];

		for (int l : label) {
			int k = l > 0 ? 0 : 1;
			this.frequencies[k]++;
			for (int i = 0; i < data[index].length; i++) {
				this.means[k][0][i] += data[index][i];
			}
			index++;
		}
		for (int k = 0; k < 2; k++) {
			for (int i = 0; i < data[0].length; i++) {
				this.means[k][0][i] = means[k][0][i] / (double) frequencies[k];
			}
		}
		this.totalNum = data.length;
		this.delta = Math.log((double) this.frequencies[0] / (double) this.frequencies[1]);
		for (int l = 0; l < 2; l++) {
			Matrix meanl = new Matrix(this.means[l]);
			Matrix theta = new Matrix(this.oge.getGraph());
			this.beta[l] = theta.times(meanl.transpose());
			this.c[l] = meanl.times(theta).times(meanl.transpose()).get(0, 0);
		}
		beta[2] = beta[0].minus(beta[1]);
	}

	public void update(double[] data, int label) {
		int k = label > 0 ? 0 : 1;
		// System.out.println(label+"\t"+k);
		this.totalNum++;
		double freq = ++this.frequencies[k];
		for (int i = 0; i < data.length; i++) {
			this.means[k][0][i] = (freq - 1.0) / freq * this.means[k][0][i] + 1.0 / freq * data[i];
		}
		this.oge.update(this.totalNum, data);
		this.delta = Math.log((double) this.frequencies[0] / (double) this.frequencies[1]);
		for (int l = 0; l < 2; l++) {
			Matrix meanl = new Matrix(this.means[l]);
			Matrix theta = new Matrix(this.oge.getGraph());
			this.beta[l] = theta.times(meanl.transpose());
			this.c[l] = meanl.times(theta).times(meanl.transpose()).get(0, 0);
		}
		beta[2] = beta[0].minus(beta[1]);
	}

	@Override
	public int predict(double[] x) {
		// TODO Auto-generated method stub
		double[][] inputArray = new double[1][x.length];
		for (int i = 0; i < x.length; i++) {
			inputArray[0][i] = x[i]-0.5*this.means[0][0][i]-0.5*this.means[1][0][i];
		}
		Matrix input = new Matrix(inputArray);
		double result = this.delta;
		// System.out.println(input.times(this.beta.get(label)).getRowDimension());
		result += input.times(this.beta[2]).get(0, 0);
		// System.out.println(result);
		return result > 0 ? 1 : -1;
	}

	@Override
	public int predict(double[] x, double[] posteriori) {
		// TODO Auto-generated method stub
		return 0;
	}

}
