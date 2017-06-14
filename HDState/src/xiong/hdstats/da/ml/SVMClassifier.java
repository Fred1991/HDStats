package xiong.hdstats.da.ml;
import ml.classification.LinearBinarySVM;
import xiong.hdstats.da.Classifier;

public class SVMClassifier implements Classifier<double[]>{
	LinearBinarySVM classifier=new LinearBinarySVM(1.0,1e-4);
	
	public SVMClassifier(double[][] d, int[] g) {
		classifier.feedData(d);
		classifier.feedLabels(g);
		classifier.train();
	}

	
	@Override
	public int predict(double[] x) {
		// TODO Auto-generated method stub
		return classifier.predict(new double[][]{x})[0];
	}

	@Override
	public int predict(double[] x, double[] posteriori) {
		// TODO Auto-generated method stub
		return this.predict(x);
	}

}
