package xiong.hdstats.opt;

import java.util.List;
import java.util.Random;

import xiong.hdstats.opt.var.ChainedMVariables;

public class GradientDescent {

	public static int GD = 1;
	public static int SGD = 2;
	public static int MBGD = 3;
	// public static int SGDFM = 4;

	public static int miniBatchSize = 2;

	public static MultiVariable getMinimum(RiskFunction f, MultiVariable intial, double eta, double tol, int maxIter,
			int ty) {
		if (!f.getClass().equals(AveragedRiskFunction.class) && ty != GD)
			return null;
		int step = 0;
		double err = Double.MAX_VALUE;
		MultiVariable val = intial.clone();
		while (step < maxIter && err > tol) {
			// MultiVariable previous = val.clone();
			MultiVariable gradient;
			if (ty == GD)
				gradient = f.gradient(val);
			else if (ty == SGD)
				gradient = ((AveragedRiskFunction) f).randomGradient(val);
			else
				gradient = ((AveragedRiskFunction) f).randomSetGradient(val, GradientDescent.miniBatchSize);
			val.updatedByGradient(gradient, eta);
			err = gradient.scalar();
			step++;
			// System.out.println(step+"\t"+err);
		}

		return val;
	}

	public static ChainedMVariables getMinimum(ChainedFunction f, ChainedMVariables intial, double eta, double tol,
			int maxIter, int ty) {
		if (!f.getClass().equals(AveragedChainedRiskFunction.class) && ty != GD)
			return null;
		int step = 0;
		double err = Double.MAX_VALUE;
		ChainedMVariables val = intial.clone();
		while (step < maxIter && (err > tol || err == 0.0)) {
			// MultiVariable previous = val.clone();
			err = 0;
			if (ty == GD) {
				do {
					MultiVariable gradient = f.gradient(val);
					val.updatedByGradient(gradient, eta);
					err += gradient.scalar();
					f.toNext();
					val.toNext();
				} while (f.innerLoop());
			} else if (ty == SGD) {
				do {
					MultiVariable gradient = ((AveragedRiskFunction) f).randomGradient(val);
					val.updatedByGradient(gradient, eta);
					err += gradient.scalar();
					f.toNext();
					val.toNext();
				} while (f.innerLoop());
				((AveragedChainedRiskFunction) f).toNextRandom();
			} else {
				do {
					MultiVariable gradient = ((AveragedChainedRiskFunction) f).randomSetGradient(val, miniBatchSize);
					val.updatedByGradient(gradient, eta);
					err += gradient.scalar();
					f.toNext();
					val.toNext();
				} while (f.innerLoop());
				((AveragedChainedRiskFunction) f).toNextRandomMiniBatch(GradientDescent.miniBatchSize);

			}
			step++;
		//	if (step % 100 == 0)
		//		System.out.println(step + "\t" + err + "\t" + f.func(val).get(0, 0));
		}
		// System.out.println(err+"\t"+f.func(val).get(0, 0));
		return val;
	}
}
