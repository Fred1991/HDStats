package edu.uva.libopt.numeric;

import java.util.Random;

import la.matrix.DenseMatrix;
import la.matrix.Matrix;

public class Utils {
	public static final int L1 = 1;
	public static final int L2 = 2;
	public static final int LINF = 4;
	public static final int LNULL = 5;



	public static double[] getGradient(NumericFunction f, double[] X, double delta) {
		double[] g = new double[X.length];
		for (int index = 0; index < g.length; index++) {
			X[index] += delta;
			double f_1 = f.func(X);
			X[index] -= delta;
			g[index] = (f_1 - f.func(X)) / delta;
		}
		return g;
	}
	
	public static double normalizedErrorL1(double[][] m1, double[][] m2, double threshold){
		double base=0;
		double err=0;
		for(int i=0;i<m1.length;i++){
			for(int j=0;j<m1[i].length;j++){
				base+=Math.abs(m1[i][j]);
				err+=Math.abs(m1[i][j]-(m2[i][j]<threshold?0:m2[i][j]));
			}
		}
		return err;//err/base;
	}	
	
	public static double normalizedErrorL2(double[][] m1, double[][] m2, double threshold){
		double base=0;
		double err=0;
		for(int i=0;i<m1.length;i++){
			for(int j=0;j<m1[i].length;j++){
				base+=m1[i][j]*m1[i][j];
				err+=((m1[i][j]-(m2[i][j]<threshold?0:m2[i][j]))*(m1[i][j]-(m2[i][j]<threshold?0:m2[i][j])));
			}
		}
		return err;//Math.sqrt(err/base);

	}

	
	public static double normalizedErrorL1Recovery(double[][] m1, double[][] m, double[][] m2, double threshold){
		double base=0;
		double err=0;
		for(int i=0;i<m1.length;i++){
			for(int j=0;j<m1[i].length;j++){
				base+=Math.abs(m1[i][j]);
				if(m2[i][j]>m[i][j])
					if(m2[i][j]<threshold&&m[i][j]==0)
						err+=Math.abs(m1[i][j]);
					else
						err+=Math.abs(m1[i][j]-m2[i][j]);
			}
		}
		return err; //err/base;
	}	
	
	public static double normalizedErrorL2Recovery(double[][] m1, double[][] m, double[][] m2, double threshold){
		double base=0;
		double err=0;
		for(int i=0;i<m1.length;i++){
			for(int j=0;j<m1[i].length;j++){
				base+=m1[i][j]*m1[i][j];
				if(m2[i][j]>m[i][j])
					if(m2[i][j]<threshold&&m[i][j]==0)
						err+=Math.abs(m1[i][j])*Math.abs(m1[i][j]);
					else
						err+=Math.abs(m1[i][j]-m2[i][j])*Math.abs(m1[i][j]-m2[i][j]);

			}
		}
		return err; //Math.sqrt(err/base);

	}

	public static void vec2Matrix(double[] vec, double[][] mat) {
		for (int i = 0; i < vec.length; i++) {
			mat[i / mat[0].length][i % mat[0].length] = vec[i];
			// System.out.println(i/mat[0].length+"\t"+i%mat[0].length+"\t"+i);
		}
	}

	public static double getLxNorm(double[] X, int l) {
		double maxX = Double.NEGATIVE_INFINITY;
		double lxnorm = 0;
		for (double x : X) {
			if (Math.abs(x) > maxX)
				maxX = Math.abs(x);
			if (l == L1)
				lxnorm += Math.abs(x);
			else if (l == L2)
				lxnorm += x * x;
			else if (l == LNULL)
				lxnorm += 0;
		}
		if (l == LINF)
			return maxX;
		else if (l == L2)
			lxnorm += Math.pow(lxnorm, 0.5);

		return lxnorm;
	}

	public static double getLxNorm(double[][] Xx, int l) {
		double maxX = Double.NEGATIVE_INFINITY;
		double lxnorm = 0;
		for (double[] X : Xx) {
			for (double x : X) {
				if (Math.abs(x) > maxX)
					maxX = Math.abs(x);
				if (l == L1)
					lxnorm += Math.abs(x);
				else if (l == L2)
					lxnorm += x * x;
				else if (l == LNULL)
					lxnorm += 0;
			}
		}
		if (l == LINF)
			return maxX;
		else if (l == L2)
			lxnorm += Math.pow(lxnorm, 0.5);

		return lxnorm;
	}

	public static double[][] getEmptyMatrix(int row, int col) {
		double[][] _mat = new double[row][col];
		return _mat;
	}

	public static double[][] getRandomMatrix(int row, int col) {
		double[][] _mat = new double[row][col];
		Random[] feeds = new Random[col];
		for (int i = 0; i < col; i++)
			feeds[i] = new Random();
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				_mat[i][j] = feeds[j].nextGaussian();
			}
		}
		return _mat;
	}

	public static double[][] getSparseRandomMatrix(int row, int col, double density) {
		double[][] _mat = new double[row][col];
		Random[] feeds = new Random[col];
		for (int i = 0; i < col; i++)
			feeds[i] = new Random();

		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (Math.random() <= density)
					_mat[i][j] = feeds[j].nextGaussian();
			}
		}
		return _mat;
	}

	public static double[][] invert(double a[][]) {
		int n = a.length;
		double x[][] = new double[n][n];
		double b[][] = new double[n][n];
		int index[] = new int[n];
		for (int i = 0; i < n; ++i)
			b[i][i] = 1;

		// Transform the matrix into an upper triangle
		gaussian(a, index);

		// Update the matrix b[i][j] with the ratios stored
		for (int i = 0; i < n - 1; ++i)
			for (int j = i + 1; j < n; ++j)
				for (int k = 0; k < n; ++k)
					b[index[j]][k] -= a[index[j]][i] * b[index[i]][k];

		// Perform backward substitutions
		for (int i = 0; i < n; ++i) {
			x[n - 1][i] = b[index[n - 1]][i] / a[index[n - 1]][n - 1];
			for (int j = n - 2; j >= 0; --j) {
				x[j][i] = b[index[j]][i];
				for (int k = j + 1; k < n; ++k) {
					x[j][i] -= a[index[j]][k] * x[k][i];
				}
				x[j][i] /= a[index[j]][j];
			}
		}
		return x;
	}

	public static void gaussian(double a[][], int index[]) {
		int n = index.length;
		double c[] = new double[n];

		// Initialize the index
		for (int i = 0; i < n; ++i)
			index[i] = i;

		// Find the rescaling factors, one from each row
		for (int i = 0; i < n; ++i) {
			double c1 = 0;
			for (int j = 0; j < n; ++j) {
				double c0 = Math.abs(a[i][j]);
				if (c0 > c1)
					c1 = c0;
			}
			c[i] = c1;
		}

		// Search the pivoting element from each column
		int k = 0;
		for (int j = 0; j < n - 1; ++j) {
			double pi1 = 0;
			for (int i = j; i < n; ++i) {
				double pi0 = Math.abs(a[index[i]][j]);
				pi0 /= c[index[i]];
				if (pi0 > pi1) {
					pi1 = pi0;
					k = i;
				}
			}

			// Interchange rows according to the pivoting order
			int itmp = index[j];
			index[j] = index[k];
			index[k] = itmp;
			for (int i = j + 1; i < n; ++i) {
				double pj = a[index[i]][j] / a[index[j]][j];

				// Record pivoting ratios below the diagonal
				a[index[i]][j] = pj;

				// Modify other elements accordingly
				for (int l = j + 1; l < n; ++l)
					a[index[i]][l] -= pj * a[index[j]][l];
			}
		}
	}

}
