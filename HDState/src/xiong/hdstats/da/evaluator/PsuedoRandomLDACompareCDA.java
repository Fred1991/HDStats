package xiong.hdstats.da.evaluator;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import edu.uva.libopt.numeric.*;
import smile.math.matrix.Matrix;
import smile.projection.PCA;
import smile.stat.distribution.SpikedMultivariateGaussianDistribution;
import smile.stat.distribution.MultivariateGaussianDistribution;
import xiong.hdstats.da.BetaLDA;
import xiong.hdstats.da.Classifier;
import xiong.hdstats.da.LDA;
import xiong.hdstats.da.PseudoInverseLDA;
import xiong.hdstats.da.OnlineLDA;
import xiong.hdstats.da.OptimalLDA;
import xiong.hdstats.da.CovLDA;
import xiong.hdstats.da.comb.OMPDA;
import xiong.hdstats.da.comb.RayleighFlowLDA;
import xiong.hdstats.da.comb.StochasticTruncatedRayleighFlowDBSDA;
import xiong.hdstats.da.comb.TruncatedRayleighFlowDBSDA;
import xiong.hdstats.da.comb.TruncatedRayleighFlowLDA;
import xiong.hdstats.da.comb.TruncatedRayleighFlowSDA;
import xiong.hdstats.da.comb.TruncatedRayleighFlowUnit;
import xiong.hdstats.da.mcmc.BayesLDA;
import xiong.hdstats.da.mcmc.LiklihoodBayesLDA;
import xiong.hdstats.da.mcmc.MCBayesLDA;
import xiong.hdstats.da.mcmc.MCRegularizedBayesLDA;
import xiong.hdstats.da.mcmc.RegularizedBayesLDA;
import xiong.hdstats.da.mcmc.RegularizedLikelihoodBayesLDA;
import xiong.hdstats.da.ml.AdaBoostTreeClassifier;
import xiong.hdstats.da.ml.AdaboostLRClassifier;
import xiong.hdstats.da.ml.DTreeClassifier;
import xiong.hdstats.da.ml.LRClassifier;
import xiong.hdstats.da.ml.NonlinearSVMClassifier;
import xiong.hdstats.da.ml.RandomForestClassifier;
import xiong.hdstats.da.ml.SVMClassifier;
import xiong.hdstats.da.shruken.DBSDA;
import xiong.hdstats.da.shruken.ODaehrLDA;
import xiong.hdstats.da.shruken.SDA;
import xiong.hdstats.da.shruken.SDABeta;
import xiong.hdstats.da.shruken.ShLDA;
import xiong.hdstats.da.shruken.ShrinkageLDA;
import xiong.hdstats.da.shruken.InvalidLDA;
import xiong.hdstats.gaussian.CovarianceEstimator;

public class PsuedoRandomLDACompareCDA {

	public static PrintStream ps = null;
	public static PrintStream ps1 = null;

	public static void main(String[] args) throws FileNotFoundException {
		for (int i = 1; i <= 10; i += 2) {
			_main(200, 10, i * 10, 500, 5);
		//	_main(200, 10, i * 10, 500, 3);
		//	_main(200, 10, i * 10, 500, 2);
		//	_main(200, 10, i * 10, 500, 1);
		}
	}

	public static void _main(int p, int nz, int initTrainSize, int testSize, int rate) throws FileNotFoundException {

//		ps = new PrintStream("C:/Users/xiongha/Desktop/beta/accuracy-" + p + "-" + nz + "-" + initTrainSize + "-"
//				+ ((double) rate / 1.0) + ".txt");
		ps1 = new PrintStream("C:/Users/xiongha/Desktop/beta/betacomp-" + p + "-" + nz + "-" + initTrainSize + "-"
				+ ((double) rate / 1.0) + ".txt");
		double[][] cov = new double[p][p];
		double[][] groupMean = new double[2][p];
		double[] mud = new double[p];

		for (int i = 0; i < cov.length; i++) {
			for (int j = 0; j < cov.length; j++) {
				cov[i][j] = Math.pow(0.8, Math.abs(i - j));
			}
		}
		double[] meanPositive = new double[p];
		double[] meanNegative = new double[p];
		for (int i = 0; i < meanPositive.length; i++) {
			if (i < nz) {
				meanPositive[i] = 1.0;
				mud[i] = 1.0;
				groupMean[0][i] = 1.0;
			} else {
				meanPositive[i] = 0.0;
				mud[i] = 0.0;
				groupMean[0][i] = 0.0;
			}
			meanNegative[i] = 0.0;
			groupMean[1][i] = 0.0;
		}

		double[][] theta_s = new Matrix(cov).inverse();
		double[] beta_s = new double[p];
		new Matrix(theta_s).ax(mud, beta_s);

		SpikedMultivariateGaussianDistribution posD = new SpikedMultivariateGaussianDistribution(meanPositive, cov);
		SpikedMultivariateGaussianDistribution negD = new SpikedMultivariateGaussianDistribution(meanNegative, cov);

		for (int r = 0; r < 100; r++) {
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
			long start = 0;
			long current = 0;

			start = System.currentTimeMillis();
			OptimalLDA opLDA = new OptimalLDA(beta_s, groupMean, -1.0 * Math.log(rate / (10.0 - rate)));
			current = System.currentTimeMillis();
//			accuracy("optimal", testData, testLabel, opLDA, start, current);

			CovarianceEstimator.lambda = 12;

			// start = System.currentTimeMillis();
			// DBSDA dbsda = new DBSDA(trainData, trainLabel, false);
			// current = System.currentTimeMillis();
			// accuracy("DBSDA", testData, testLabel, dbsda, start, current);

			start = System.currentTimeMillis();
			SDABeta sda = new SDABeta(trainData, trainLabel, false);
			current = System.currentTimeMillis();
//			accuracy("SDA", testData, testLabel, sda, start, current);
			betacompare("SDA", sda.getBeta(), beta_s);

//			accuracy("LDA", testData, testLabel, LDA, start, current);

			for (int i = 5; i < 15; i++) {
				start = System.currentTimeMillis();
				OMPDA olda = new OMPDA(trainData, trainLabel, false, i);
				current = System.currentTimeMillis();
	//			accuracy("OMP-" + (i), testData, testLabel, olda, start, current);
				betacompare("OMP-" + (i), olda.getBeta(), beta_s);
			}

			// for (int i = 1; i < 20; i++) {
			// start = System.currentTimeMillis();
			// TruncatedRayleighFlowUnit olda = new
			// TruncatedRayleighFlowUnit(trainData, trainLabel, false, i * 2 +
			// 2);
			// current = System.currentTimeMillis();
			// accuracy("TruncatedRayleighFlowUnit-" + (i), testData, testLabel,
			// olda, start, current);
			// betacompare("TruncatedRayleighFlowUnit-" +
			// (i),olda.getBeta(),beta_s);
			// }

			for (int i = 5; i < 15; i++) {
				start = System.currentTimeMillis();
				TruncatedRayleighFlowLDA olda = new TruncatedRayleighFlowLDA(trainData, trainLabel, false, i);
				current = System.currentTimeMillis();
		//		accuracy("TruncatedRayleighFlowLDA-" + i, testData, testLabel, olda, start, current);
				betacompare("TruncatedRayleighFlowLDA-" + i, olda.getBeta(), beta_s);
			}

			// Estimator.lambda=12;

			// for (int i = 1; i < 20; i++) {
			// start = System.currentTimeMillis();
			// TruncatedRayleighFlowSDA olda = new
			// TruncatedRayleighFlowSDA(trainData, trainLabel, false, i * 2 +
			// 2);
			// current = System.currentTimeMillis();
			// accuracy("TruncatedRayleighFlowSDA-" + (i), testData, testLabel,
			// olda, start, current);
			// betacompare("TruncatedRayleighFlowSDA-" +
			// (i),olda.getBeta(),beta_s);
			// }

			for (int i = 5; i < 15; i++) {
				start = System.currentTimeMillis();
				TruncatedRayleighFlowDBSDA olda = new TruncatedRayleighFlowDBSDA(trainData, trainLabel, false, i);
				current = System.currentTimeMillis();
			//	accuracy("TruncatedRayleighFlowDBSDA-" + i, testData, testLabel, olda, start, current);
				betacompare("TruncatedRayleighFlowDBSDA-" + i, olda.getBeta(), beta_s);
			}

			// for (int i = 0; i < 5; i++) {
			// start = System.currentTimeMillis();
			// TruncatedRayleighFlowDBSDA olda = new
			// TruncatedRayleighFlowDBSDA(trainData, trainLabel, false,
			// i * 2 + 6);
			// current = System.currentTimeMillis();
			// accuracy("TruncatedRayleighFlowDBSDA-" + (i * 2 + 6), testData,
			// testLabel, olda, start, current);
			// }

			start = System.currentTimeMillis();
			RayleighFlowLDA olda = new RayleighFlowLDA(trainData, trainLabel, false);
			current = System.currentTimeMillis();
		//	accuracy("RayleighFlowLDA", testData, testLabel, olda, start, current);
			betacompare("RayleighFlowLDA", olda.getBeta(), beta_s);

			// start = System.currentTimeMillis();
			// DBSDA dblda = new DBSDA(trainData, trainLabel, false);
			// current = System.currentTimeMillis();
			// accuracy("DBSDA", testData, testLabel, dblda, start, current);

			// Estimator.lambda = 32.0;

			// System.out.println(Utils.getErrorInf(olda.means[0], new
			// double[1][p]));

		}
	}

	private static void betacompare(String name, double[] beta, double[] betas) {
		// int[] plabels=new int[labels.length];
		// System.out.println("accuracy statistics");
		if (beta.length != betas.length)
			System.exit(-1);
		System.out.println(beta.length + "\t" + betas.length);
		double betasL2 = Utils.getLxNorm(betas, Utils.L2);
		double betaL2 = Utils.getLxNorm(beta, Utils.L2);

		int tp = 0, fp = 0, tn = 0, fn = 0;
		double[] err = new double[beta.length];
		for (int i = 0; i < beta.length; i++) {
			// System.out.println(beta[i] + "\t vs\t" + betas[i]);
			if (Math.abs(beta[i]) >= 1e-10 && Math.abs(betas[i]) >= 1e-10) {
				tp++;
			} else if (Math.abs(beta[i]) < 1e-10 && Math.abs(betas[i]) < 1e-10) {
				tn++;
			} else if (Math.abs(beta[i]) >= 1e-10 && Math.abs(betas[i]) < 1e-10) {
				fp++;
			} else {
				fn++;
			}
			err[i] = beta[i]/betaL2 - betas[i]/betasL2;

		}
		ps1.println(name + "\t" + tp + "\t" + tn + "\t" + fp + "\t" + fn + "\t" + Utils.getLxNorm(err, Utils.L1) + "\t"
				+ Utils.getLxNorm(err, Utils.L2) + "\t" + Utils.getLxNorm(beta, Utils.L0)+ "\t" + Utils.getLxNorm(beta, Utils.L2));

	}

	private static void accuracy(String name, double[][] data, int[] labels, Classifier<double[]> classifier, long t1,
			long t2) {
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
		ps.println(name + "\t" + tp + "\t" + tn + "\t" + fp + "\t" + fn + "\t" + train_time + "\t" + test_time);

	}
}
