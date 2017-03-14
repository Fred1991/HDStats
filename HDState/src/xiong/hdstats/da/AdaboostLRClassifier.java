package xiong.hdstats.da;

import ml.classification.AdaBoost;
import ml.classification.LogisticRegression;

public class AdaboostLRClassifier implements Classifier<double[]>{
	AdaBoost classifier=null;
	
	public AdaboostLRClassifier(double[][] d, int[] g, int num) {
		LogisticRegression[] lrs=new LogisticRegression[num];
		for(int i=0;i<num;i++)
			lrs[i]=new LogisticRegression(1,0.1);
		classifier=new AdaBoost(lrs);
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
