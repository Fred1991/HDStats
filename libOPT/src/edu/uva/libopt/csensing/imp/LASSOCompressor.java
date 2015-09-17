package edu.uva.libopt.csensing.imp;

import edu.uva.libopt.csensing.Compressor;
import edu.uva.libopt.numeric.NumericFunction;
import edu.uva.libopt.numeric.NumericOptimizer;
import edu.uva.libopt.numeric.Utils;
import edu.uva.libopt.numeric.optimizer.SparseGradientOptimizer;
import la.matrix.DenseMatrix;
import la.matrix.Matrix;

public class LASSOCompressor implements Compressor {

	private double _lambda = 1.0;
	private NumericOptimizer _opt = null;
	public double epsilon=0.001;
	public double delta=0.0001;

	public static class LossFunction implements NumericFunction {

		private Matrix _dict;
		private double[] _vec;

		public LossFunction(double[][] dict, double[] vec) {
			_dict = new DenseMatrix(dict);
			_vec = vec;

		}

		@Override
		public double func(double[] X) {
			// TODO Auto-generated method stub
			Matrix vecx = new DenseMatrix(X, 1);
			Matrix _vecx = _dict.mtimes(vecx);
			double error_l2=0;
			for (int i = 0; i < _vec.length; i++) {
				error_l2 +=Math.pow(_vecx.getData()[i][0]-_vec[i],2);
			}
			return error_l2;
		}
		
		@Override
		public double[] gradient(double[] X) {
			// TODO Auto-generated method stub
			return Utils.getGradient(this, X, 0.000001);
		}

	}

	public LASSOCompressor(double lambda) {
		this._lambda = lambda;
		this._opt = new SparseGradientOptimizer(Utils.L1, this._lambda);
	}

	@Override
	public double[] compress(double[][] dic, double[] vector) {
		// TODO Auto-generated method stub
		Matrix dict = new DenseMatrix(dic);
		// System.out.println(dict.getRowDimension()+" x "+dict.getColumnDimension());

		Matrix vecx = new DenseMatrix(vector, 1);
		// System.out.println(vecx.getRowDimension()+" x "+vecx.getColumnDimension());

		Matrix _vecx = dict.mtimes(vecx);
		// System.out.println(_vecx.getRowDimension()+" x "+_vecx.getColumnDimension());

		double[] output = new double[dict.getRowDimension()];
		for (int i = 0; i < output.length; i++) {
			output[i] = _vecx.getData()[i][0];
		}

		return output;
	}

	@Override
	public double decompress(double[][] dic, double[] vector, double[] output) {
		// TODO Auto-generated method stub
		LossFunction lfunc=new LossFunction(dic,vector);
		return this._opt.getMinimum(output, epsilon, delta, lfunc);
	}

}
