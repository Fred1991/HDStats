package xiong.hdstats.da;

import Jama.Matrix;

public class BetaLDA implements Classifier<double[]> {
	public double[][][] means;
	public int[] frequencies = new int[2];
	public double[] c = new double[2];
	public Matrix[] beta = new Matrix[3];
	public double delta = 0;
	public int totalNum;

	public void init(double[][] data, double[][] graph, int[] label) {
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
			Matrix theta = new Matrix(graph);
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
		double result = -1.0*this.delta;
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
