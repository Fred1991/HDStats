package xiong.hdstats.opt.estimator;

import Jama.Matrix;
import xiong.hdstats.opt.GradientDescent;
import xiong.hdstats.opt.var.MatrixMVariable;
import xiong.hdstats.opt.MultiVariable;
import xiong.hdstats.opt.RiskFunction;

public class Lasso implements RiskFunction {
	public Matrix X;
	public Matrix Y;
	public double lambda;

	public Lasso(Matrix _X, Matrix _Y, double _l) {
		this.X = _X;
		this.Y = _Y;
		this.lambda = _l;
	}

	@Override
	public Matrix func(MultiVariable input) {
		// TODO Auto-generated method stub
		MatrixMVariable mmv = (MatrixMVariable) input;
		double value = this.Y.minus(this.X.times(mmv.getMtx())).normF() + mmv.getMtx().norm1() * this.lambda;
		Matrix m = new Matrix(1, 1);
		m.set(0, 0, value);
		return m;
	}

	@Override
	public MultiVariable gradient(MultiVariable input) {
		// TODO Auto-generated method stub
		Matrix mtx = ((MatrixMVariable) input).getMtx();
		Matrix gradient = X.transpose().times(-1).times(Y.minus(X.times(mtx)));
		//System.out.println(gradient.getArray()[0].length);
		for (int i = 0; i < mtx.getArray().length; i++) {
			for (int j = 0; j < mtx.getArray()[0].length; j++) {
				if (mtx.getArray()[i][j] > 0)
					gradient.getArray()[i][j] += this.lambda;
				else if (mtx.getArray()[i][j] < 0)
					gradient.getArray()[i][j] -= this.lambda;

			}
		}
	//	gradient = gradient.times(1.0/gradient.norm2());
		return new MatrixMVariable(gradient);
	}

	@Override
	public MultiVariable project(MultiVariable input) {
		// TODO Auto-generated method stub
		return input;
	}
	
	public static void main(String[] args) {
		Matrix truth = Matrix.random(1000, 1);
		for (int i = 0; i < 100; i++)
			truth.getArray()[i][0] = 0;
		Matrix _X = Matrix.random(300, 1000);
		Matrix _Y = _X.times(truth);
		Matrix _init = new Matrix(1000, 1);
		for (double l = 0.3; l < 0.4; l += 0.01) {
			Lasso lasso = new Lasso(_X, _Y, l);
			MatrixMVariable result = (MatrixMVariable) GradientDescent.getMinimum(lasso, new MatrixMVariable(_init),
					1e-4, 1e-3, 10000, GradientDescent.GD);
			System.out.println(l+"\t"+result.getMtx().minus(truth).normF() / truth.normF());
		}
	}

}
