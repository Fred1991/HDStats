package xiong.hdstats.da.comb;

import java.util.HashMap;

public class MetricTransfomer {

	public static double[][] getFeatureSelectedData(double[] beta, double[][] data) {
		int index = 0;
		HashMap<Integer, Integer> mapping = new HashMap<Integer, Integer>();
		for (int i = 0; i < beta.length; i++) {
			if (beta[i] != 0 || Math.abs(beta[i]) < 10e-10) {
				mapping.put(index++, i);
			}
		}
		int numDim = mapping.size();
		double[][] _data = new double[data.length][numDim];
		for (int i = 0; i < data.length; i++) {
			for (int j = 0; j < data[i].length; j++) {
				_data[i][j] = data[i][mapping.get(j)];
			}
		}
		return _data;
	}

	public static double[][] getMetricTransformedData(double[] beta, double[][] data) {
		int index = 0;
		HashMap<Integer, Integer> mapping = new HashMap<Integer, Integer>();
		for (int i = 0; i < beta.length; i++) {
			if (beta[i] != 0 || Math.abs(beta[i]) < 10e-10) {
				mapping.put(index++, i);
			}
		}
		int numDim = mapping.size();
		double[][] _data = new double[data.length][numDim];
		for (int i = 0; i < data.length; i++) {
			for (int j = 0; j < data[i].length; j++) {
				_data[i][j] = data[i][mapping.get(j)] * beta[mapping.get(j)];
			}
		}
		return _data;
	}
}
