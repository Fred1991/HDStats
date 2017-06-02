package xiong.hdstats.da;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import Jama.Matrix;
import smile.stat.distribution.GLassoMultivariateGaussianDistribution;
import smile.stat.distribution.MultivariateGaussianDistribution;
import xiong.hdstats.MLEstimator;

public class StochasticRayleighFlow {
	private double eta;
	private double[][] I;
	private Matrix vector;
	private Matrix AMat;
	private Matrix BMat;
	private int p;
	private double noise;
	private GLassoMultivariateGaussianDistribution mgd;
	public StochasticRayleighFlow(int p, double eta, double noise) {
		this.eta = eta;
		this.noise=noise;
		I = new double[p][p];
		for (int i = 0; i < p; i++)
			I[i][i] = 1.0;
		this.p = p;
		double[] noiseMean=new double[p];
		mgd=new GLassoMultivariateGaussianDistribution(noiseMean,I);
	}

	public StochasticRayleighFlow(double eta, double[][] cov, double noise) { // as PCA
		this(cov.length, eta, noise);
		double[][] arrayB = new double[p][p];
		for (int i = 0; i < p; i++)
			arrayB[i][i] = 1.0;
		double[][] arrayA = cov;
		this.setAB(arrayA, arrayB);
	}

	public StochasticRayleighFlow(double eta, double[][] cov1, double[][] cov2, double noise) { // as																// LDA
		this(cov1.length, eta, noise);
		double[][] arrayB = cov2;
		double[][] arrayA = cov1;
		this.setAB(arrayA, arrayB);
	}

	public void setAB(double[][] A, double[][] B) {
		this.setA(A);
		this.setB(B);
	}

	private void setA(double[][] A) {
		AMat = new Matrix(A);
	}

	private void setB(double[][] B) {
		BMat = new Matrix(B);
	}

	public void init(double[] vector) {
		this.vector = new Matrix(vector, 1).transpose();
	}

	public void iterateWithBatchUpdate(double[][] newData) {
		MLEstimator MLE = new MLEstimator();
		Matrix _AMat = new Matrix(MLE.covariance(newData));
		Matrix _AMat_ = this.AMat;
		this.AMat = _AMat;
		this.iterate();
		this.AMat = _AMat_;
	}

	public Matrix getVector() {
		return this.vector;
	}

	public void iterate() {
		// System.out.println(vector.getRowDimension()+"\t"+vector.getColumnDimension());
		double rho = vector.transpose().times(AMat).times(vector).get(0, 0);
		rho /= vector.transpose().times(BMat).times(vector).get(0, 0);
		Matrix C = AMat.minus(BMat.times(rho)).times(eta / rho).plus(new Matrix(I));
		Matrix vUpdate = C.times(vector);
		vUpdate = vUpdate.times(1.0 / vUpdate.normF());
		double[] vArray = vUpdate.transpose().getArray()[0];
		double[] noiseVec=l2Normalized(mgd.rand());
 		for(int i=0;i<p;i++)
			vArray[i]+=noiseVec[i];
		vArray = l2Normalized(vArray);
		this.vector = new Matrix(vArray, 1).transpose();
	}

	private static double[] l2Normalized(double[] v) {
		double norml2 = 0;
		for (double vv : v)
			norml2 += vv * vv;
		norml2 = Math.sqrt(norml2);
		for (int i = 0; i < v.length; i++) {
			v[i] /= norml2;
		}
		return v;
	}

}
