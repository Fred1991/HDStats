package edu.uva.hdstats.da;
import smile.classification.*;
import smile.math.kernel.*;

public class NonlinearSVMClassifier implements Classifier<double[]>{
	SVM<double[]> classifier = null;
	
	public NonlinearSVMClassifier(double[][] d, int[] g, double sigma, double softM) {
		classifier=new SVM<double[]>(new GaussianKernel(sigma),softM);
		this.classifier.learn(d, g);
	}

	
	@Override
	public int predict(double[] x) {
		// TODO Auto-generated method stub
		return classifier.predict(x);
	}

	@Override
	public int predict(double[] x, double[] posteriori) {
		// TODO Auto-generated method stub
		return this.predict(x);
	}

}
