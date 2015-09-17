package edu.uva.libopt.csensing.imp;

import java.util.List;

import edu.uva.libopt.csensing.DictionaryLearner;
import edu.uva.libopt.numeric.NumericFunction;
import edu.uva.libopt.numeric.NumericOptimizer;
import edu.uva.libopt.numeric.Utils;
import edu.uva.libopt.numeric.optimizer.GradientDescentOptimizer;

public class LazyLassoLearner implements DictionaryLearner{
	public double epsilon=0.0001;
	public double delta=0.00001;
	private double _lambda=0;
	private NumericOptimizer _opt;

	public LazyLassoLearner(double lambda){
		this._lambda=lambda;
		this._opt=new GradientDescentOptimizer();

	}
	
	
	public class LazyLossFunction implements NumericFunction{
		private List<double[]> _vex;
		private int _row;
		private int _col;
		private double[][] _mat;
		private LASSOCompressor _lasso;

		public LazyLossFunction(List<double[]> vectors, int row, int col, double lambda){
			this._vex=vectors;
			this._row=row;
			this._col=col;
			this._mat=Utils.getEmptyMatrix(_row, _col);
			this._lasso=new LASSOCompressor(_lambda);
		}

		@Override
		public double func(double[] X) {
			// TODO Auto-generated method stub
			Utils.vec2Matrix(X, _mat);
			double error=0;
			for(double[] vec:_vex){
				double[] dcpvec=new double[_col];
				error+=this._lasso.decompress(_mat, vec, dcpvec);
				error+=(_lambda*Utils.getLxNorm(dcpvec, Utils.L1));
			}
			return error;
		}
		@Override
		public double[] gradient(double[] X) {
			// TODO Auto-generated method stub
			return Utils.getGradient(this, X, 0.000001);
		}
	}

	@Override
	public double[][] learn(List<double[]> vectors, int row, int col) {
		// TODO Auto-generated method stub
		LazyLossFunction lfunc=new LazyLossFunction(vectors,row,col,_lambda);
		double[] coeff=Utils.getRandomMatrix(1, row*col)[0];
		this._opt.getMinimum(coeff, epsilon, delta, lfunc);
		double[][] rMat=Utils.getEmptyMatrix(row, col);
		Utils.vec2Matrix(coeff, rMat);
		return rMat;
	}

}
