package xiong.hdstats.preceptor;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import Jama.Matrix;

public class MIMO {
	public Matrix weights;
	public Matrix intercepts;
	public Matrix connectivity;
	public Matrix inputSample;

	public double eta;
	public Random rand;
	public static double alpha = 10;
	public int inNum;
	public int outNum;

	public static List<MIMO> buildNetwork(int dim, int nClass, int numInnerLayer, int dimInnerLayer, int degree) {
		List<MIMO> network = new ArrayList<MIMO>();
		network.add(new MIMO(dim,dimInnerLayer,10e-4,degree));
		for(int i=0;i<numInnerLayer;i++){
			network.add(new MIMO(dimInnerLayer,dimInnerLayer,10e-4,degree));
		}
		network.add(new MIMO(dimInnerLayer,nClass,10e-4));
		return network;
	}

	public static Matrix output(List<MIMO> network, Matrix input) {
		Matrix outVec = null;
		for (int i = 0; i < network.size(); i++) {
			outVec = network.get(i).output(outVec);
		}
		return outVec;
	}

	public static class GradientMIMO {
		public Matrix gradientW;
		public Matrix gradientI;
	}

	public MIMO(int input, int output, double eta) {
		this.weights = Matrix.random(output, input);
		this.intercepts = Matrix.random(output, 1);
		this.inputSample = Matrix.random(input, 1);
		this.rand = new Random();
		this.inNum = input;
		this.outNum = output;
	}

	public MIMO(int input, int output, double eta,
			int degree) { /* randomized solution */
		this(input, output, eta);
		if(degree == -1)
			return;
		this.connectivity = new Matrix(output, input);
		for (int i = 0; i < input; i++) {
			int filled = 0;
			while (filled < degree) {
				int j = rand.nextInt(output);
				if (this.connectivity.get(j, i) == 0.0) {
					this.connectivity.set(j, i, 1.0);
					filled++;
				}
			}
		}

	}

	private double delta(double x) {
		double ex = Math.exp(alpha * x);
		return 2 * alpha * ex / (1 + ex) / (1 + ex);
	}

	public GradientMIMO gradientInnerLayerAt(Matrix inVec) {
		Matrix outVec = this.output(inVec);
		GradientMIMO gmm = new GradientMIMO();
		Matrix gW = new Matrix(this.weights.getRowDimension(), this.weights.getColumnDimension());
		Matrix gI = new Matrix(this.weights.getRowDimension(), 1);
		for (int i = 0; i < outNum; i++) {
			for (int j = 0; j < inNum; j++) {
				gW.set(i, j, this.delta(outVec.get(i, 0)) * inVec.get(j, 0));
			}
			gI.set(i, 0, this.delta(outVec.get(i, 0)));
		}
		gmm.gradientW = gW;
		gmm.gradientI = gI;
		return gmm;
	}

	public void updateInputSample(Matrix inVec) {
		this.inputSample = this.inputSample.times(1 - eta).plus(inVec.times(eta));
	}

	public void updateThisLayer(GradientMIMO gmm) {
		updateThisLayer(gmm.gradientW, gmm.gradientI);
	}

	public void updateThisLayer(Matrix gradientW, Matrix gradientI) {
		this.weights = this.weights.plus(gradientW.times(-1 * eta));
		if (this.connectivity != null) {
			for (int i = 0; i < connectivity.getRowDimension(); i++) {
				for (int j = 0; i < connectivity.getColumnDimension(); j++) {
					if (connectivity.get(i, j) == 0) {
						this.weights.set(i, j, 0);
					}
				}
			}
		}
		this.intercepts = this.intercepts.plus(gradientI.times(-1 * eta));
	}

	private Matrix outputImp(Matrix inVec) {
		return weights.times(inVec).plus(intercepts);
	}

	public Matrix output(Matrix input) {
		Matrix toOutput = this.outputImp(input);
		for (int i = 0; i < toOutput.getArray().length; i++) {
			for (int j = 0; i < toOutput.getArray()[i].length; j++) {
			//	if (toOutput.getArray()[i][j] >= 0) {
			//		toOutput.getArray()[i][j] = 1;
			//	} else {
			//		toOutput.getArray()[i][j] = -1;
			//	}
				double ex = Math.exp(-1*alpha*toOutput.getArray()[i][j]);
				double density = (1-ex)/(1+ex);
				toOutput.getArray()[i][j] = density;
			}
		}
		return toOutput;
	}
}
