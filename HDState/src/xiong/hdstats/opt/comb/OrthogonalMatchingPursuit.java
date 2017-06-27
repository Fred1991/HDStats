package xiong.hdstats.opt.comb;

import Jama.Matrix;
import xiong.hdstats.opt.GradientDescent;
import xiong.hdstats.opt.MultiVariable;
import xiong.hdstats.opt.RiskFunction;
import xiong.hdstats.opt.estimator.LADLasso;
import xiong.hdstats.opt.var.MatrixMVariable;

public class OrthogonalMatchingPursuit implements RiskFunction {
	public Matrix X;
	public Matrix Y;
	public int nonzeros;
	
	public static int getL0Norm(Matrix _m){
		int nz = 0;
		for(int i=0;i<_m.getArray().length;i++){
			for(int j=0;j<_m.getArray()[i].length;j++){
				if(_m.getArray()[i][j]!=0)
					nz++;
			}
		}
		return nz;
	}

	public OrthogonalMatchingPursuit(Matrix _X, Matrix _Y, int _l) {
		this.X = _X;
		this.Y = _Y;
		this.nonzeros = _l;
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
		Matrix gradient = new Matrix(mtx.getArray().length, mtx.getArray()[0].length);
		Matrix c = Y.minus(X.times(mtx));
		for (int k = 0; k < c.getArray().length; k++) {
			for (int l = 0; l < mtx.getArray().length; l++) {
				for (int n = 0; n < mtx.getArray()[0].length; n++) {
					if (c.getArray()[k][n] > 0)
						gradient.getArray()[l][n] -= X.getArray()[k][l];
					else if (c.getArray()[k][n] < 0)
						gradient.getArray()[l][n] += X.getArray()[k][l];
				}
			}
		}

		return new MatrixMVariable(gradient);
	}
	
	@Override
	public MultiVariable project(MultiVariable input) {
		// TODO Auto-generated method stub
		return input;
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