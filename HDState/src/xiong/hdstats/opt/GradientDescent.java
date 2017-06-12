package xiong.hdstats.opt;

public class GradientDescent {

	public static int GD = 1;
	public static int SGD = 2;
	public static int MBGD = 3;

	public static int miniBatchSize = 2;

	public static MultiVariable getMinimum(RiskFunction f, MultiVariable intial, double eta, double tol, int maxIter,
			int ty) {
		if (!f.getClass().equals(AveragedRiskFunction.class) && ty != GD)
			return null;
		int step = 0;
		double err = Double.MAX_VALUE;
		MultiVariable val = intial.clone();
		while (step < maxIter && err > tol) {
		//	MultiVariable previous = val.clone();
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
	//		System.out.println(step+"\t"+err);
		}

		return val;
	}

}
