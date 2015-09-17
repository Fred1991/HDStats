package edu.uva.libopt.numeric.optimizer;

import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import ml.optimization.AcceleratedGradientDescent;
import ml.optimization.AcceleratedProximalGradient;
import ml.optimization.Prox;
import ml.optimization.ProxL1;
import ml.optimization.ProxL2;
import ml.optimization.ProxL2Square;
import ml.optimization.ProxLInfinity;
import ml.optimization.ProxPlus;
import edu.uva.libopt.numeric.NumericFunction;
import edu.uva.libopt.numeric.NumericOptimizer;
import edu.uva.libopt.numeric.Utils;

public class SparseGradientOptimizer implements NumericOptimizer {


	private int norm;
	private double lambda;

	public SparseGradientOptimizer(int norm, double lambda) {
		this.norm = norm;
		this.lambda = lambda;
	}

	@Override
	public double getMinimum(double[] X, double epsilon, double delta,
			NumericFunction f) {
		// TODO Auto-generated method stub

		Matrix W = new DenseMatrix(X, 2); // Initial matrix (vector) you want to
											// optimize
		// initialize optimizer
		if (norm == Utils.L1)
			AcceleratedProximalGradient.prox = new ProxL1(lambda);
		else if (norm == Utils.L2)
			AcceleratedProximalGradient.prox = new ProxL2(lambda);
		else if (norm == Utils.LINF)
			AcceleratedProximalGradient.prox = new ProxLInfinity(lambda);
		else
			AcceleratedProximalGradient.prox = new Prox();

		Matrix G = new DenseMatrix(f.gradient(X), 2); // Gradient
																		// at
																		// the
																		// initial
																		// matrix
																		// (vector)
																		// you
																		// want
																		// to
																		// optimize

		double fval = f.func(X);
		double hval = ((double)Utils.getLxNorm(X, norm));

		boolean flags[] = null;
		while (true) {
			flags = AcceleratedProximalGradient.run(G, fval, hval, epsilon, W); // Update
																			// W
																			// in
																			// place
			if (flags[0]) // flags[0] indicates if it converges
				break;

			fval = f.func(W.getData()[0]);
			hval = ((double)Utils.getLxNorm(X, norm));
			
			
		//	System.out.println(hval);
			
			if (flags[1]) // flags[1] indicates if gradient at the updated W is
							// required
				G = new DenseMatrix(
						f.gradient(W.getData()[0]), 2); // Compute
																			// the
																			// gradient
																			// at
																			// the
																			// new
																			// W

		}

//		System.out.println("weights:" + W.getData()[0][0] + "\t"
//				+ W.getData()[0][1]);
		return f.func(X);
	}

}
