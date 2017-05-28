package xiong.hdstats.da;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import edu.uva.libopt.numeric.*;
import smile.math.matrix.Matrix;
import smile.projection.PCA;
import smile.stat.distribution.GLassoMultivariateGaussianDistribution;
import xiong.hdstats.Estimator;
import xiong.hdstats.da.AdaBoostTreeClassifier;
import xiong.hdstats.da.AdaboostLRClassifier;
import xiong.hdstats.da.BayesLDA;
import xiong.hdstats.da.Classifier;
import xiong.hdstats.da.DTreeClassifier;
import xiong.hdstats.da.GLassoLDA;
import xiong.hdstats.da.LDA;
import xiong.hdstats.da.LRClassifier;
import xiong.hdstats.da.LiklihoodBayesLDA;
import xiong.hdstats.da.MCBayesLDA;
import xiong.hdstats.da.MCRegularizedBayesLDA;
import xiong.hdstats.da.NonSparseLDA;
import xiong.hdstats.da.NonlinearSVMClassifier;
import xiong.hdstats.da.ODaehrLDA;
import xiong.hdstats.da.OLDA;
import xiong.hdstats.da.OnlineLDA;
import xiong.hdstats.da.OrgLDA;
import xiong.hdstats.da.PDLassoLDA;
import xiong.hdstats.da.RandomForestClassifier;
import xiong.hdstats.da.RegularizedBayesLDA;
import xiong.hdstats.da.RegularizedLikelihoodBayesLDA;
import xiong.hdstats.da.SVMClassifier;
import xiong.hdstats.da.ShLDA;
import xiong.hdstats.da.ShrinkageLDA;
import xiong.hdstats.da.mDaehrLDA;

public class PsuedoRandomOnlineLDA {

	public static PrintStream ps = null;

	public static void main(String[] args) throws FileNotFoundException {
		_main(200, 10, 200, 500, 5000, 5);
	}

	public static void _main(int p, int nz, int initTrainSize, int testSize, int newSize, int rate)
			throws FileNotFoundException {

		ps = System.out;
		double[][] cov = new double[p][p];

		for (int i = 0; i < cov.length; i++) {
			for (int j = 0; j < cov.length; j++) {
				cov[i][j] = Math.pow(0.8, Math.abs(i - j));
			}
		}
		double[] meanPositive = new double[p];
		double[] meanNegative = new double[p];
		for (int i = 0; i < meanPositive.length; i++) {
			if (i < nz)
				meanPositive[i] = 1.0;
			else
				meanPositive[i] = 0.0;
			meanNegative[i] = 0.0;
		}

		double[][] theta_s = new Matrix(cov).inverse();
		GLassoMultivariateGaussianDistribution posD = new GLassoMultivariateGaussianDistribution(meanPositive, cov);

		GLassoMultivariateGaussianDistribution negD = new GLassoMultivariateGaussianDistribution(meanNegative, cov);

		for (int r = 0; r < 20; r++) {
			double[][] testData = new double[testSize][p];
			int[] testLabel = new int[testSize];
			for (int i = 0; i < testSize; i++) {
				double[] tdat;
				if (i % 10 < rate) {
					tdat = posD.rand();
					testLabel[i] = 1;
				} else {
					tdat = negD.rand();
					testLabel[i] = -1;
				}
				for (int j = 0; j < cov.length; j++)
					testData[i][j] = tdat[j];
			}

			double[][] trainData = new double[initTrainSize][p];
			int[] trainLabel = new int[initTrainSize];
			for (int i = 0; i < initTrainSize; i++) {
				double[] tdat;
				{
					if (i % 10 < rate) {
						tdat = posD.rand();
						trainLabel[i] = 1;
					} else {
						tdat = negD.rand();
						trainLabel[i] = -1;
					}
					for (int j = 0; j < cov.length; j++)
						trainData[i][j] = tdat[j];
				}

			}
			L0NormLDAClassifier olda = new L0NormLDAClassifier(trainData, trainLabel,false,50);
			// System.out.println(Utils.getErrorInf(olda.means[0], new
			// double[1][p]));
			accuracy("Oline-init", 0, testData, testLabel, olda, 0, 0);
		}
	}

	private static void accuracy(String name, int s, double[][] data, int[] labels, Classifier<double[]> classifier,
			long t1, long t2) {
		// int[] plabels=new int[labels.length];
		// System.out.println("accuracy statistics");
		int tp = 0, fp = 0, tn = 0, fn = 0;
		for (int i = 0; i < labels.length; i++) {
			int pl = classifier.predict(data[i]);
			// System.out.println(pl + "\t vs\t" + labels[i]);
			if (pl == 1 && labels[i] == 1) {
				tp++;
			} else if (pl == -1 && labels[i] == -1) {
				tn++;
			} else if (pl == 1 && labels[i] == -1) {
				fp++;
			} else {
				fn++;
			}

		}
		long train_time = t2 - t1;
		double test_time = ((double) (System.currentTimeMillis() - t2)) / ((double) labels.length);
		ps.println(
				name + "\t" + s + "\t" + tp + "\t" + tn + "\t" + fp + "\t" + fn + "\t" + train_time + "\t" + test_time);

	}

	private static void plotAccuracy(double[][] fm, double[][] fm2, HashMap<Integer, Set<Integer>> missingcodes,
			int sum_miss_codes, double[][] fm3, int sum_recover_codes, double threshold) {
		HashMap<Integer, Set<Integer>> recoverycodes = new HashMap<Integer, Set<Integer>>();

		for (int i = 0; i < fm.length; i++) {
			for (int j = 0; j < fm[i].length; j++) {
				if (fm2[i][j] == 0 && fm3[i][j] > threshold) {
					sum_recover_codes++;
					if (!recoverycodes.containsKey(i))
						recoverycodes.put(i, new HashSet<Integer>());
					recoverycodes.get(i).add(j);

				}
			}
		}

		int intersection = intersection(missingcodes, recoverycodes);
		double precision = (double) intersection / (double) sum_recover_codes;
		double recall = (double) intersection / (double) sum_miss_codes;
		double f1 = 2 * precision * recall / (precision + recall);
		double err_l1 = Utils.normalizedErrorL1Recovery(fm, fm2, fm3, threshold);
		double err_l2 = Utils.normalizedErrorL2Recovery(fm, fm2, fm3, threshold);

		ps.print(threshold);
		ps.print("," + sum_miss_codes);
		ps.print("," + sum_recover_codes);
		ps.print("," + intersection);
		ps.print("," + f1);
		ps.print("," + precision);
		ps.print("," + recall);
		ps.print("," + err_l1);
		ps.println("," + err_l2);

	}

	public static double F1(Set<String> miss, Set<String> recover) {
		int intersect = 0;
		for (String i : miss)
			if (recover.contains(i))
				intersect++;
		return 2.0 * (double) intersect / (double) (miss.size() + recover.size());
	}

	public static int intersection(HashMap<Integer, Set<Integer>> miss, HashMap<Integer, Set<Integer>> recover) {
		int intersect = 0;
		for (int p : miss.keySet()) {
			if (recover.containsKey(p)) {
				for (int code : miss.get(p)) {
					if (recover.get(p).contains(code)) {
						intersect++;
					}
				}
			}
		}

		return intersect;
	}

}
