package xiong.hdstats.opt;

import Jama.Matrix;

public interface RiskFunction {
	public Matrix func(MultiVariable input);
	public MultiVariable gradient(MultiVariable input);
	public MultiVariable project(MultiVariable input);
}
