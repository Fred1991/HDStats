package edu.uva.libopt.numeric.optimizer;

import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import ml.optimization.AcceleratedGradientDescent;
import edu.uva.libopt.numeric.NumericFunction;
import edu.uva.libopt.numeric.NumericOptimizer;
import edu.uva.libopt.numeric.Utils;

public class GradientDescentOptimizer implements NumericOptimizer {

	@Override
	public double getMinimum(double[] X, double epsilon, double delta,
			NumericFunction f) {
		// TODO Auto-generated method stub

		Matrix W = new DenseMatrix(X, 2); // Initial matrix (vector) you want to
											// optimize
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

		boolean flags[] = null;
		while (true) {
			flags = AcceleratedGradientDescent.run(G, fval, epsilon, W); // Update
																			// W
																			// in
																			// place
			if (flags[0]) // flags[0] indicates if it converges
				break;
			fval = f.func(W.getData()[0]);
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

		System.out.println("weights:" + W.getData()[0][0] + "\t"
				+ W.getData()[0][1]);
		return f.func(X);
	}

}
