package xiong.hdstats.da;

import smile.classification.*;
public class AdaBoostTreeClassifier implements Classifier<double[]>{
	AdaBoost classifier= null;
	
	public AdaBoostTreeClassifier(double[][] d, int[] g, int nTree) {
		this.classifier= new AdaBoost(d, g, nTree);
	}


	@Override
	public int predict(double[] x) {
		// TODO Auto-generated method stub
		return this.classifier.predict(x);
	}

	@Override
	public int predict(double[] x, double[] posteriori) {
		// TODO Auto-generated method stub
		return this.classifier.predict(x,posteriori);
	}

}
