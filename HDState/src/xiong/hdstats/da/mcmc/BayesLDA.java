package xiong.hdstats.da.mcmc;

import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import edu.uva.libopt.numeric.Utils;
import gov.sandia.cognition.math.RingAccumulator;
import gov.sandia.cognition.math.matrix.Matrix;
import gov.sandia.cognition.math.matrix.MatrixFactory;
import gov.sandia.cognition.math.matrix.Vector;
import gov.sandia.cognition.math.matrix.VectorFactory;
import gov.sandia.cognition.math.matrix.mtj.DenseMatrixFactoryMTJ;
import gov.sandia.cognition.math.matrix.mtj.decomposition.CholeskyDecompositionMTJ;
import gov.sandia.cognition.statistics.distribution.InverseWishartDistribution;
import gov.sandia.cognition.statistics.distribution.MultivariateGaussian;
import smile.stat.distribution.MultivariateGaussianDistribution;
import xiong.hdstats.NearPD;
import xiong.hdstats.da.PseudoInverseLDA;

public class BayesLDA extends PseudoInverseLDA {

	private List<double[][]> pooledClassifiers = new ArrayList<double[][]>();
	private HashMap<double[][], MultivariateGaussianDistribution> scoreFunctions = new HashMap<double[][], MultivariateGaussianDistribution>();
	private HashMap<double[][], Double> weights = new HashMap<double[][], Double>();
	private double[][] precision;
	private int numPredictors = -1;

	public static Matrix sample(final Random random, final Vector mean, final Matrix covarianceSqrt,
			final int degreesOfFreedom) {
		ArrayList<Vector> xs = MultivariateGaussian.sample(mean, covarianceSqrt, random, degreesOfFreedom);
		RingAccumulator<Matrix> sum = new RingAccumulator<Matrix>();
		for (Vector x : xs) {
			sum.accumulate(x.outerProduct(x));
		}
		return sum.getSum();
	}

	public BayesLDA(double[][] d, int[] g, int size, double K) {
		super(d, g, false);
		VectorFactory vf = VectorFactory.getDenseDefault();
		MatrixFactory mf = MatrixFactory.getDenseDefault();
		// double[][] cov =
		// mf.copyArray(this.pooledInverseCovariance).inverse().toArray();
		// for (int i = 0; i < pooledInverseCovariance.length; i++) {
		// for (int j = 0; j < pooledInverseCovariance.length; j++) {
		// if (i == j) {
		// pooledInverseCovariance[i][j] += K;
		// }
		// }
		// }
		this.precision = this.pooledInverseCovariance;
		double cov[][] = mf.copyArray(super.pooledInverseCovariance).inverse().toArray();
		for (int i = 0; i < cov.length; i++) {
			for (int j = 0; j < cov.length; j++) {
				if (i == j) {
					cov[i][j] += K;
				}
			}
		}
		NearPD npd=new NearPD();
		npd.calcNearPD(new Jama.Matrix(cov));
		Vector meanV = vf.copyArray(super.globalMean);
		Matrix covarianceSqrt = CholeskyDecompositionMTJ.create(
				DenseMatrixFactoryMTJ.INSTANCE.copyMatrix(mf.copyArray(npd.getX().getArray())))
				.getR();
		int fDOF = Math.min(Math.max(this.precision.length, g.length) * 10, 2000);
		Random r = new Random(System.currentTimeMillis());
		InverseWishartDistribution iwd=new InverseWishartDistribution(mf.copyArray(super.pooledInverseCovariance),
																		Math.max(this.pooledInverseCovariance.length+1, g.length-1));  
		while (pooledClassifiers.size() < size) {
			Matrix mtx = sample(r, meanV, covarianceSqrt, fDOF);
			// NonSparseEstimator nse=new NonSparseEstimator();
			Matrix m=mtx.inverse();
			double[][] marray = m.toArray();
			pooledClassifiers.add(marray);
			MultivariateGaussianDistribution gaussian = new MultivariateGaussianDistribution(super.globalMean, marray,
					false);
			scoreFunctions.put(marray, gaussian);
			weights.put(marray, iwd.getProbabilityFunction().logEvaluate(m));
			System.out.println("sampled\t" + pooledClassifiers.size() + "\t matrices");

			// double weight =
			// sampler.getProbabilityFunction().logEvaluate(mtx.inverse());
			// if (Math.exp(weight) > 0 && Math.exp(weight) < 1) {
			//
			// sum_prob+=Math.exp(weight);
			// }
			//
		}
		// System.out.println("sum of log\t"+sum_log_prob);
		// for (double[][] icov : weights.keySet()) {
		// weights.put(icov, weights.get(icov) / sum_prob);
		// }
		//
	}

	private double score(double[][] xs, double[][] lda) {
		return scoreFunctions.get(lda).logLikelihood(xs);
	}

	private double score(double[] x, double[][] lda) {
		return scoreFunctions.get(lda).logp(x);
	}

	public void setNumPredictors(int np) {
		this.numPredictors = np;
	}

	@Override
	public int predict(double[] x) {
		// TODO Auto-generated method stub
		HashMap<Integer, HashMap<double[][], Double>> probX = new HashMap<Integer, HashMap<double[][], Double>>();
		HashMap<Integer, Double> maxValues = new HashMap<Integer, Double>();
		HashMap<Integer, Double> sumExpLogProbs = new HashMap<Integer, Double>();
		HashMap<Integer, Double> scores = new HashMap<Integer, Double>();
		List<double[][]> obtainedClassifiers;
		if (numPredictors < 0) {
			obtainedClassifiers = pooledClassifiers;
		} else {
			obtainedClassifiers = new ArrayList<double[][]>();
			while (obtainedClassifiers.size() < numPredictors) {
				obtainedClassifiers.add(pooledClassifiers.get(obtainedClassifiers.size()));
			}
		}

		for (double[][] pm : obtainedClassifiers) {
			super.pooledInverseCovariance = pm;
			int label = super.predict(x);

			double s = score(x, pm) + weights.get(pm);
			if (!probX.containsKey(label)) {
				probX.put(label, new HashMap<double[][], Double>());
				maxValues.put(label, Double.NEGATIVE_INFINITY);
			}
			// else {
			probX.get(label).put(pm, s);
			// }
			// probX.put(pm, s);
			if (s > maxValues.get(label))
				maxValues.put(label, s);
		}
		for (int label : probX.keySet()) {
			for (double[][] pm : probX.get(label).keySet()) {
				if (!sumExpLogProbs.containsKey(label))
					sumExpLogProbs.put(label, 0.0);
				sumExpLogProbs.put(label,
						sumExpLogProbs.get(label) + (Math.exp(probX.get(label).get(pm) - maxValues.get(label))));
			}
		}
		for (int label : maxValues.keySet()) {
			scores.put(label, maxValues.get(label) + Math.log(sumExpLogProbs.get(label)));
			System.out.println(scores.get(label));

		}

		double maxScore = Double.NEGATIVE_INFINITY;
		int maxScoreLabel = Integer.MIN_VALUE;
		for (int label : scores.keySet()) {
			if (scores.get(label) >= maxScore) {
				maxScore = scores.get(label);
				maxScoreLabel = label;
			}
		}
		if (maxScoreLabel == Integer.MIN_VALUE) {
			int index = (int) (Math.random() * scores.keySet().size());
			return new ArrayList<Integer>(scores.keySet()).get(index);
		}
		return maxScoreLabel;
	}

	/**
	 * Test case with the original values from the tutorial of Kardi Teknomo
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		int[] group = { 1, 1, 1, 1, 2, 2, 2 };
		double[][] data = { { 2.95, 6.63 }, { 2.53, 7.79 }, { 3.57, 5.65 }, { 3.16, 5.47 }, { 2.58, 4.46 },
				{ 2.16, 6.22 }, { 3.27, 3.52 } };

		BayesLDA test = new BayesLDA(data, group, 1000, 1);
		double[] testData = { 3.57, 4.46 };

		// test
		double[] values = test.getDiscriminantFunctionValues(testData);
		for (int i = 0; i < values.length; i++) {
			System.out.println("Discriminant function " + (i + 1) + ": " + values[i]);
		}

		System.out.println("Predicted group: " + test.predict(testData));
	}

}
