package xiong.hdstats.da.ml;
import smile.classification.*;
import xiong.hdstats.da.Classifier;

public class KNNClassifier implements Classifier<double[]>{
	KNN<double[]> classifier = null;
	
	public KNNClassifier(double[][] d, int[] g, int k) {
		classifier=KNN.learn(d, g, k);
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
