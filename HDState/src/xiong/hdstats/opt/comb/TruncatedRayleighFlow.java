package xiong.hdstats.opt.comb;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import Jama.Matrix;
import edu.uva.libopt.numeric.Utils;
import xiong.hdstats.gaussian.SampleCovarianceEstimator;

public class TruncatedRayleighFlow {
	private int k;
	private double eta;
	private double[][] I;
	private Matrix vector;
	private Matrix AMat;
	private Matrix BMat;
	private int p;
	private double tol;

	public TruncatedRayleighFlow(int k, int p, double eta, double tol) {
		this.k = k;
		this.eta = eta;
		I = new double[p][p];
		for (int i = 0; i < p; i++)
			I[i][i] = 1.0;
		this.p = p;
		this.tol =tol;
	}

	public TruncatedRayleighFlow(int k, double eta, double tol, double[][] cov) { // as PCA
		this(k, cov.length, eta, tol);
		double[][] arrayB = new double[p][p];
		for (int i = 0; i < p; i++)
			arrayB[i][i] = 1.0;
		double[][] arrayA = cov;
		this.setAB(arrayA, arrayB);
	}

	public TruncatedRayleighFlow(int k, double eta, double tol, double[][] cov1, double[][] cov2) { // as
																									// LDA
		this(k, cov1.length, eta, tol);
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
		SampleCovarianceEstimator MLE = new SampleCovarianceEstimator();
		Matrix _AMat = new Matrix(MLE.covariance(newData));
		Matrix _AMat_ = this.AMat;
		this.AMat = _AMat;
		this.iterate();
		this.AMat = _AMat_;
	}

	public Matrix getVector() {
		return this.vector;
	}

	public boolean iterate() {
		// System.out.println(vector.getRowDimension()+"\t"+vector.getColumnDimension());
		double rho = vector.transpose().times(AMat).times(vector).get(0, 0);
		rho /= vector.transpose().times(BMat).times(vector).get(0, 0);
		Matrix C = AMat.minus(BMat.times(rho)).times(eta / rho).plus(new Matrix(I));
		Matrix vUpdate = C.times(vector);
		vUpdate = vUpdate.times(1.0 / vUpdate.normF());
		double[] vArray = vUpdate.transpose().getArray()[0];
		vArray = truncate(k, vArray);
		vArray = l2Normalized(vArray);
		double[] err =new double[vArray.length];
		for(int i=0;i<err.length;i++){
			err[i] = vArray[i]-this.vector.get(i,0);
		}
		this.vector = new Matrix(vArray, 1).transpose();
		if(Utils.getLxNorm(err, Utils.LINF)>=(tol*eta)){
			return true;
		}
		else{
			return false;
		}
	}

	private static double[] truncate(int k, double[] v) {
		List<Double> items = new ArrayList<Double>();
		for (double vv : v) {
			items.add(Math.abs(vv));
		}
		Collections.sort(items);
		double thr = items.get(items.size() - k);
		for (int i = 0; i < v.length; i++) {
			if (Math.abs(v[i]) < thr) {
				v[i] = 0.0;
			}
		}
		return v;
	}

	private static double[] l2Normalized(double[] v) {
		double norml2 = 0;
		for (double vv : v)
			norml2 += vv * vv;
		norml2 = Math.sqrt(norml2);
		for (int i = 0; i < v.length; i++) {
			v[i] = v[i] / norml2;
		}
		return v;
	}

}
