package xiong.hdstats.opt.estimator.MF;

import Jama.Matrix;

public class MFUtil {

	public static int L1 = 1;
	public static int L2 = 2;
	public static int LInf = 3;
	public static int normal = 1;
	public static int nmf = 2;
	public static int prob = 3;

	public static void subMatrixSelection(Matrix m, int[][] selection) {
		for (int i = 0; i < selection.length; i++) {
			for (int j = 0; j < selection[0].length; j++) {
				if (selection[i][j] == 0)
					 m.getArray()[i][j] = 0;
			}
		}
	}

	public static Matrix getL2NormGradient(Matrix m) {
		return m.times(2);
	}

	public static Matrix getL1NormGradient(Matrix m) {
		double[][] aCopy = m.getArrayCopy();
		for (int i = 0; i < aCopy.length; i++) {
			for (int j = 0; j < aCopy[0].length; j++) {
				if (aCopy[i][j] >= 0)
					aCopy[i][j] = 1;
				else
					aCopy[i][j] = -1;
			}
		}
		return new Matrix(aCopy);
	}

	public static Matrix getLInfNormGradient(Matrix m) {
		double[][] aCopy = m.getArrayCopy();
		for (int i = 0; i < aCopy.length; i++) {
			for (int j = 0; j < aCopy[0].length; j++) {
				if (aCopy[i][j] >= 0)
					aCopy[i][j] = 1;
				else
					aCopy[i][j] = -1;
			}
		}
		return new Matrix(aCopy);
	}

	public static Matrix getEmptyGradient(Matrix m) {
		double[][] aCopy = m.getArrayCopy();
		for (int i = 0; i < aCopy.length; i++) {
			for (int j = 0; j < aCopy[0].length; j++) {
				aCopy[i][j] = 0;
			}
		}
		return new Matrix(aCopy);
	}

	public static Matrix getLpNormGradient(Matrix m, int p) {
		if (p == L1)
			return getL1NormGradient(m);
		else if (p == L2)
			return getL2NormGradient(m);
		else
			return getLInfNormGradient(m);
	}

	public static void nonnegativeHT(Matrix m) {
		for (int i = 0; i < m.getArray().length; i++) {
			for (int j = 0; j < m.getArray()[0].length; j++) {
				if (m.getArray()[i][j] < 0)
					m.getArray()[i][j] = 0;
			}
		}
	}

	public static void probHT(Matrix m) {
		for (int i = 0; i < m.getArray().length; i++) {
			for (int j = 0; j < m.getArray()[0].length; j++) {
				if (m.getArray()[i][j] < 0)
					m.getArray()[i][j] = 0;
				else if (m.getArray()[i][j] > 1)
					m.getArray()[i][j] = 1;
			}
		}
	}

}
