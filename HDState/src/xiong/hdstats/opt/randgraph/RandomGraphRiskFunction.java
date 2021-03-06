package xiong.hdstats.opt.randgraph;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import Jama.Matrix;
import xiong.hdstats.opt.AveragedRiskFunction;
import xiong.hdstats.opt.MultiVariable;
import xiong.hdstats.opt.RiskFunction;

public class RandomGraphRiskFunction implements RiskFunction {
	public List<RiskFunction> funcs = new ArrayList<RiskFunction>();
	public RandomGraph rn;

	public RandomGraphRiskFunction(AveragedRiskFunction arf, RandomGraph rg) {
		for (RiskFunction f : arf.funcs) {
			this.addFunction(f);
		}
		rn = rg;
		rn.randStart();
	}

	public RandomGraphRiskFunction(Collection<RiskFunction> fs, RandomGraph rg) {
		for (RiskFunction f : fs) {
			this.addFunction(f);
		}
		rn = rg;
		rn.randStart();
	}

	public void addFunction(RiskFunction func) {
		this.funcs.add(func);
	}

	@Override
	public Matrix func(MultiVariable input) {
		// TODO Auto-generated method stub
		if (funcs.size() < 1)
			return new Matrix(0, 0);
		Matrix m = this.funcs.get(0).func(input).copy();
		for (int i = 1; i < funcs.size(); i++) {
			m.plus(this.funcs.get(i).func(input).copy());
		}
		return m.times(1.0 / funcs.size());
	}

	@Override
	public MultiVariable gradient(MultiVariable input) {
		// TODO Auto-generated method stub
		if (funcs.size() < 1)
			return null;
		MultiVariable m = this.funcs.get(0).gradient(input).clone();
		for (int i = 1; i < funcs.size(); i++) {
			m = m.plus(this.funcs.get(i).gradient(input).clone());
		}
		return m.times(1.0 / funcs.size());
	}

	public MultiVariable randomGradient(MultiVariable input) {
		int n = funcs.size();
		int i = Math.abs(rn.nextInt()) % n;
		return this.funcs.get(i).gradient(input);
	}

	public MultiVariable randomSetGradient(MultiVariable input, int miniSize) {
		Set<Integer> miniBatch = new HashSet<Integer>();
		int n = funcs.size();
		while (miniBatch.size() < miniSize) {
			int i = Math.abs(rn.nextInt()) % n;
			miniBatch.add(i);
		}

		if (miniBatch.size() < 1)
			return null;
		List<Integer> miniBatchList = new ArrayList<Integer>(miniBatch);
		MultiVariable m = this.funcs.get(miniBatchList.get(0)).gradient(input).clone();
		for (int i = 1; i < miniBatchList.size(); i++) {
			m = m.plus(this.funcs.get(miniBatchList.get(i)).gradient(input).clone());
		}
		return m.times(1.0 / miniBatchList.size());
	}

	@Override
	public MultiVariable project(MultiVariable input) {
		// TODO Auto-generated method stub
		int n = funcs.size();
		int i = Math.abs(rn.nextInt()) % n;
		return this.funcs.get(i).gradient(input);
	}
}
