package xiong.hdstats.opt.var;

import edu.uva.libopt.numeric.Utils;
import xiong.hdstats.opt.MultiVariable;

public class ArrayMVariable implements MultiVariable {
	private double[] array;

	public ArrayMVariable(double[] _a) {
		this.array = _a;
	}

	public double[] getArray() {
		return this.array;
	}

	@Override
	public void updatedByGradient(MultiVariable gradient, double eta) {
		// TODO Auto-generated method stub
		for (int i = 0; i < array.length; i++) {
			this.array[i]-=((ArrayMVariable)gradient).array[i]*eta;
		}
	}

	@Override
	public MultiVariable clone() {
		// TODO Auto-generated method stub
		double[] exist = new double[this.array.length];
		for (int i = 0; i < array.length; i++)
			exist[i] = array[i];
		return new ArrayMVariable(exist);
	}

	@Override
	public MultiVariable plus(MultiVariable m) {
		// TODO Auto-generated method stub
		ArrayMVariable amv = (ArrayMVariable) this.clone();
		for (int i = 0; i < amv.array.length; i++) {
			amv.array[i] += ((ArrayMVariable) m).array[i];
		}
		return amv;
	}

	@Override
	public MultiVariable times(double d) {
		// TODO Auto-generated method stub
		ArrayMVariable amv = (ArrayMVariable) this.clone();
		for (int i = 0; i < amv.array.length; i++) {
			amv.array[i] *= d;
		}
		return amv;
	}

	@Override
	public double scalar() {
		// TODO Auto-generated method stub
		return Utils.getLxNorm(array, Utils.L1);
	}

}
