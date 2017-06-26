package xiong.hdstats.opt;

import java.util.ArrayList;
import java.util.List;

import Jama.Matrix;

public class ChainedRiskFunction implements ChainedFunction {
	public List<RiskFunction> funcs = new ArrayList<RiskFunction>();
	public int index = 0;

	public RiskFunction getCurrent() {
		return this.funcs.get(index);
	}

	public boolean innerLoop() {
		if (index % funcs.size() != 0)
			return true;
		else
			return false;
	}

	public ChainedRiskFunction(List<RiskFunction> crf) {
		for (RiskFunction c : crf)
			funcs.add(c);
	}

	@Override
	public Matrix func(MultiVariable input) {
		// TODO Auto-generated method stub
		return funcs.get(index).func(input);
	}

	@Override
	public MultiVariable gradient(MultiVariable input) {
		// TODO Auto-generated method stub
		return funcs.get(index).gradient(input);
	}

	public void toNext() {
		index++;
		index = index % funcs.size();
	}

	@Override
	public MultiVariable project(MultiVariable input) {
		// TODO Auto-generated method stub
		return input;
	}
}
