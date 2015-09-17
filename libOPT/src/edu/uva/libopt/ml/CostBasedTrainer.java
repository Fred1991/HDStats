package edu.uva.libopt.ml;

import edu.uva.libopt.numeric.NumericFunction;
import edu.uva.libopt.numeric.NumericOptimizer;
import edu.uva.libopt.numeric.Utils;
import edu.uva.libopt.numeric.optimizer.GradientDescentOptimizer;
import edu.uva.libopt.numeric.optimizer.SparseGradientOptimizer;

public class CostBasedTrainer implements NumericFunction {

	private CostBasedLearner learner = null;
	public double epsilon = 0.00001;
	public double delta = 0.00001;

	public CostBasedTrainer(CostBasedLearner learner) {
		this.learner = learner;
	}


	@Override
	public double func(double[] X) {
		// TODO Auto-generated method stub
		return this.learner.costFunction(X);
	}

	public void trainSparseParams(int lnorm, double ratio, double[] params_init) {
		NumericOptimizer nopt = new SparseGradientOptimizer(lnorm, ratio);
		nopt.getMinimum(params_init, epsilon, delta, this);
	}

	public void trainDenseParams(double[] params_init) {
		NumericOptimizer nopt = new GradientDescentOptimizer();
		nopt.getMinimum(params_init, epsilon, delta, this);
	}
	
	@Override
	public double[] gradient(double[] X) {
		// TODO Auto-generated method stub
		return Utils.getGradient(this, X, 0.000001);
	}

}
