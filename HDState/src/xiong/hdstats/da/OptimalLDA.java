package xiong.hdstats.da;

import Jama.Matrix;

public class OptimalLDA implements Classifier<double[]> {
	public double[][][] means;
	public Matrix beta;
	public double delta = 0;
	public int totalNum;

	public OptimalLDA(double[] beta, double[][] gmeans, double delta) {
		means = new double[2][1][gmeans[0].length];
		for(int i=0;i<2;i++){
			for(int j=0;j<gmeans[i].length;j++){
				means[i][0][j]=gmeans[i][j];
			}
		}
		this.beta =new Matrix(beta,1).transpose();
		this.delta=delta;
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
		result += input.times(this.beta).get(0, 0);
		// System.out.println(result);
		return result > 0 ? 1 : -1;
	}

	@Override
	public int predict(double[] x, double[] posteriori) {
		// TODO Auto-generated method stub
		return 0;
	}

}
