package xiong.hdstats.opt.comb;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import Jama.Matrix;
import xiong.hdstats.opt.GradientDescent;
import xiong.hdstats.opt.MultiVariable;
import xiong.hdstats.opt.RiskFunction;
import xiong.hdstats.opt.var.MatrixMVariable;

public class OrthogonalMatchingPursuit implements RiskFunction {
	public Matrix X;
	public Matrix Y;
	public int nonzeros;

	public static int getL0Norm(Matrix _m) {
		int nz = 0;
		for (int i = 0; i < _m.getArray().length; i++) {
			for (int j = 0; j < _m.getArray()[i].length; j++) {
				if (_m.getArray()[i][j] != 0)
					nz++;
			}
		}
		return nz;
	}

	public OrthogonalMatchingPursuit(Matrix _X, Matrix _Y, int _nzeros) {
		this.X = _X;
		this.Y = _Y;
		this.nonzeros = _nzeros;
	}

	@Override
	public Matrix func(MultiVariable input) {
		// TODO Auto-generated method stub
		MatrixMVariable mmv = (MatrixMVariable) input;
		double value = this.Y.minus(X.times(mmv.getMtx())).norm1() + getL0Norm(mmv.getMtx());
		Matrix m = new Matrix(1, 1);
		m.set(0, 0, value);
		return m;
	}

	@Override
	public MultiVariable gradient(MultiVariable input) {
		// TODO Auto-generated method stub
		Matrix mtx = ((MatrixMVariable) input).getMtx();
		Matrix gradient = X.transpose().times(-1).times(Y.minus(X.times(mtx)));
		return new MatrixMVariable(gradient);
	}

	@Override
	public MultiVariable project(MultiVariable input) {
		// TODO Auto-generated method stub
		hardThreshold(((MatrixMVariable) input).getMtx().getArray(), this.nonzeros);
		return input;
	}

	public static void hardThreshold(double[][] mtx, int k) {
		List<Double> values = new ArrayList<Double>();
		for (int i = 0; i < mtx.length; i++) {
			for (int j = 0; j < mtx[i].length; j++) {
				values.add(Math.abs(mtx[i][j]));
			}
		}
		Collections.sort(values);
		double thr = values.get(values.size() - k);
		for (int i = 0; i < mtx.length; i++) {
			for (int j = 0; j < mtx[i].length; j++) {
				if (Math.abs(mtx[i][j]) < thr)
					mtx[i][j] = 0;
			}
		}
	}

	public static void main(String[] args) {
		Matrix truth = Matrix.random(100, 1);
		for (int i = 5; i < 100; i++)
			truth.getArray()[i][0] = 0;
		Matrix _X = Matrix.random(50, 100);
		Matrix _Y = _X.times(truth);
		Matrix _init = new Matrix(100, 1);
		// for (double m = 1; m < 10; m += 0.1) {
		OrthogonalMatchingPursuit OMP = new OrthogonalMatchingPursuit(_X, _Y, 20);
		MatrixMVariable result = (MatrixMVariable) GradientDescent.getMinimum(OMP, new MatrixMVariable(_init),
				1e-4, 1e-3, 10000, GradientDescent.GD);
		System.out.println(result.getMtx().minus(truth).normF() / truth.normF()+"\t"+getL0Norm(result.getMtx()));
		// }
	}
}