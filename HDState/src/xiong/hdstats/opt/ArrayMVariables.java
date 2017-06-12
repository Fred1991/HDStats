package xiong.hdstats.opt;

import edu.uva.libopt.numeric.Utils;

public class ArrayMVariables implements MultiVariable {
	private double[] array;

	public ArrayMVariables(double[] _a) {
		this.array = _a;
	}

	public double[] getArray() {
		return this.array;
	}

	@Override
	public void updatedByGradient(MultiVariable gradient, double eta) {
		// TODO Auto-generated method stub
		for (int i = 0; i < array.length; i++) {
			this.array[i]-=((ArrayMVariables)gradient).array[i]*eta;
		}
	}

	@Override
	public MultiVariable clone() {
		// TODO Auto-generated method stub
		double[] exist = new double[this.array.length];
		for (int i = 0; i < array.length; i++)
			exist[i] = array[i];
		return new ArrayMVariables(exist);
	}

	@Override
	public MultiVariable plus(MultiVariable m) {
		// TODO Auto-generated method stub
		ArrayMVariables amv = (ArrayMVariables) this.clone();
		for (int i = 0; i < amv.array.length; i++) {
			amv.array[i] += ((ArrayMVariables) m).array[i];
		}
		return amv;
	}

	@Override
	public MultiVariable times(double d) {
		// TODO Auto-generated method stub
		ArrayMVariables amv = (ArrayMVariables) this.clone();
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
