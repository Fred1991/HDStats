package xiong.hdstats.da;

import smile.classification.*;
public class RandomForestClassifier implements Classifier<double[]>{
	RandomForest classifier= null;
	
	public RandomForestClassifier(double[][] d, int[] g, int nTree) {
		this.classifier= new RandomForest(d, g, nTree);
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
