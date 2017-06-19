package xiong.hdstats.opt.estimator;

import Jama.Matrix;
import xiong.hdstats.opt.GradientDescent;
import xiong.hdstats.opt.var.MatrixMVariable;
import xiong.hdstats.opt.MultiVariable;
import xiong.hdstats.opt.RiskFunction;

public class LADLasso implements RiskFunction{
	public Matrix X;
	public Matrix Y;
	public double lambda;
	
	public LADLasso(Matrix _X, Matrix _Y, double _l) {
		this.X = _X;
		this.Y = _Y;
		this.lambda = _l;
	}
	
	@Override
	public Matrix func(MultiVariable input) {
		// TODO Auto-generated method stub
		MatrixMVariable mmv = (MatrixMVariable) input;
		double value = this.Y.minus(X.times(mmv.getMtx())).norm1() + mmv.getMtx().norm1();
		Matrix m = new Matrix(1, 1);
		m.set(0, 0, value);
		return m;
	}
	
	@Override
	public MultiVariable gradient(MultiVariable input) {
		// TODO Auto-generated method stub
		Matrix mtx = ((MatrixMVariable) input).getMtx();
		//Matrix gradient = X.transpose().times(-1).times(Y.minus(X.times(mtx)));
		Matrix gradient = new Matrix(mtx.getArray().length,mtx.getArray()[0].length);
		//int[] a = IntStream.range(0,(mtx.getArray().length - 1)).toArray();
		Matrix c = Y.minus(X.times(mtx));
		//System.out.println(c.getArray().length);
		//System.out.println(c.getArray()[0].length);
		//System.out.println(mtx.getArray().length);
		//System.out.println(mtx.getArray()[0].length);
		//System.out.println(gradient.getArray().length);
		//System.out.println(gradient.getArray()[0].length);
		
		/*for (int k = 0; k < c.getArray().length; k++){
			for (int l = 0; l < mtx.getArray().length; l++){
				if (c.getArray()[k][mtx.getArray()[0].length-1] > 0)
					gradient.getArray()[l][mtx.getArray()[0].length-1] -= X.getArray()[k][l];
				else if (c.getArray()[k][mtx.getArray()[0].length-1] < 0)
					gradient.getArray()[l][mtx.getArray()[0].length-1] += X.getArray()[k][l];
			}
		}*/
		
		for (int k = 0; k < c.getArray().length; k++){
			for (int l = 0; l < mtx.getArray().length; l++){
				for (int n = 0; n < mtx.getArray()[0].length; n++){
					if (c.getArray()[k][n] > 0)
						gradient.getArray()[l][n] -= X.getArray()[k][l];
					else if (c.getArray()[k][n] < 0)
						gradient.getArray()[l][n] += X.getArray()[k][l];
				}
			}
		}
		
		for (int i = 0; i < mtx.getArray().length; i++) {
			for (int j = 0; j < mtx.getArray()[0].length; j++) {
				if (mtx.getArray()[i][j] > 0)
					gradient.getArray()[i][j] += this.lambda;
				else if (mtx.getArray()[i][j] < 0)
					gradient.getArray()[i][j] -= this.lambda;

			}
		}
		return new MatrixMVariable(gradient);
	}

	public static void main(String[] args) {
		Matrix truth = Matrix.random(100, 1);
		for (int i = 10; i < 100; i++)
			truth.getArray()[i][0] = 0;
		Matrix _X = Matrix.random(50, 100);
		Matrix _Y = _X.times(truth);
		Matrix _init = new Matrix(100, 1);
		for (double m = 1; m < 10; m += 0.1) {
			LADLasso LADlasso = new LADLasso(_X, _Y, m);
			MatrixMVariable result = (MatrixMVariable) GradientDescent.getMinimum(LADlasso, new MatrixMVariable(_init),
					1e-4, 1e-3, 10000, GradientDescent.GD);
			System.out.println(m+"\t"+result.getMtx().minus(truth).normF() / truth.normF());
		}
	}
}
