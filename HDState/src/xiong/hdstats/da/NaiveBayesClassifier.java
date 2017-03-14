package xiong.hdstats.da;
import smile.classification.*;

public class NaiveBayesClassifier implements Classifier<double[]>{
	NaiveBayes classifier = null;
	
	public NaiveBayesClassifier(double[][] d, int[] g) {
		classifier=new NaiveBayes(NaiveBayes.Model.GENERAL,2,d[0].length);
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
