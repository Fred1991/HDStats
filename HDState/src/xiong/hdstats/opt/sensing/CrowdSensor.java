package xiong.hdstats.opt.sensing;

import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import Jama.Matrix;
import xiong.hdstats.opt.AveragedChainedRiskFunction;
import xiong.hdstats.opt.ChainedFunction;
import xiong.hdstats.opt.GradientDescent;
import xiong.hdstats.opt.estimator.MF.LpMF;
import xiong.hdstats.opt.estimator.MF.MFUtil;
import xiong.hdstats.opt.var.ChainedMVariables;

public class CrowdSensor {
	public static PrintStream ps;
	public static int cycle = 0;
	public static List<double[]> allData;
	public static HashMap<Integer, CrowdSensor> cmap = new HashMap<Integer, CrowdSensor>();
	public static double[][] estimated;
	public static double[][] centraRes;
	public static double[][] CFRes;

	public static double[][] truth;

	public int id;
	public HashMap<Integer, Set<Integer>> collected = new HashMap<Integer, Set<Integer>>();
	public double[][] collectedData;
	public int[][] nz;

	public CrowdSensor() {
		this.id = cmap.size();
		cmap.put(id, this);
	}

	public static void creatWorldWithCrowds(String fname, int N) {
		cycle = 0;
		cmap = new HashMap<Integer, CrowdSensor>();
		allData = DataLoader.allSensorData(fname);
		for (int i = 0; i < N; i++)
			new CrowdSensor();
	}

	public static void pseudoLocations(double r, int maxLocations) {
		Set<CrowdSensor> selected = new HashSet<CrowdSensor>();
		while (selected.size() < cmap.size() * r) {
			int index = (int) (Math.random() * cmap.size());
			index = index == cmap.size() ? index - 1 : index;
			selected.add(cmap.get(index));
		}
		for (CrowdSensor cs : selected) {
			int num = (int) Math.random() * maxLocations;
			num = num == 0 ? 1 : num;
			// num=50;
			Set<Integer> ca = new HashSet<Integer>();
			cs.collected.put(cycle, ca);
			while (ca.size() < num) {
				ca.add((int) (Math.random() * allData.size()));
			}
		}
	}

	public static void autoMasking(double maxNoise) {
		double[][] _addData = DataLoader.getAllDataBeforeTime(allData, cycle + 1);
		truth = _addData;
		for (CrowdSensor cs : cmap.values()) {
			cs.collectedData = new double[_addData.length][_addData[0].length];
			cs.nz = new int[_addData.length][_addData[0].length];
			for (int t : cs.collected.keySet()) {
				for (int a : cs.collected.get(t)) {
					cs.collectedData[t][a] = _addData[t][a] * (1 + maxNoise * (Math.random() - 0.5));
					cs.nz[t][a] = 1;
				}
			}
		}
	}

	public static void main(String[] args) throws FileNotFoundException {
		ps = new PrintStream("C:\\Users\\xiongha\\Desktop\\tempOutput3.txt");
		for (int maxLoc = 1; maxLoc <= 5; maxLoc++)
			for (int crowdSize = 10; crowdSize <= 30; crowdSize += 5) {
				for (int par = 10; par <= 10; par++) {
					for (int wind = 20; wind <= 50; wind += 10) {
						for (int latent = 2; latent <= 10; latent += 2) {
							creatWorldWithCrowds("C:\\Users\\xiongha\\Desktop\\tempLeye.csv", crowdSize);
							List<Double> absErrorCSWA = new ArrayList<Double>();
							List<Double> absErrorCentra = new ArrayList<Double>();

							for (; cycle < 100; cycle++) {
								pseudoLocations(((double) par) / 10.0, maxLoc);
								autoMasking(0.01);
								if (cycle > wind) {
									runCSWA(wind, latent, absErrorCSWA);
									runCentraCS(wind, latent, absErrorCentra);

								}
							}
							plot("CSWA",maxLoc, crowdSize, par, wind, latent, absErrorCSWA);
							plot("centraCS",maxLoc, crowdSize, par, wind, latent, absErrorCSWA);

						}
					}
				}
			}
	}

	private static void plot(String name, int maxLoc, int crowdSize, int par, int wind, int latent, List<Double> absErrorCSWA) {
		double aMSE = 0;
		for (double m : absErrorCSWA)
			aMSE += m;
		aMSE /= absErrorCSWA.size();

		ps.println(name+"\t" + latent + "\t" + wind + "\t" + crowdSize + "\t" + par
				+ "\t" + maxLoc + "\t" + aMSE);
	}

	private static void runCSWA(int wind, int latent, List<Double> absErrorCSWA) {
		estimating(0.01, 0.01, wind, latent);
		double absE = 0;

		for (int i = 0; i < estimated[estimated.length - 1].length; i++) {
			double err = Math.abs(estimated[estimated.length - 1][i] - truth[truth.length - 1][i]);
			System.out.println(estimated[estimated.length - 1][i] + "\t" + truth[truth.length - 1][i]);
			absE += err / estimated[estimated.length - 1].length;
		}
		absErrorCSWA.add(absE);
	}
	
	private static void runCentraCS(int wind, int latent, List<Double> absErrorCSWA) {
		centraEstimate(0.01, 0.01, wind, latent);
		double absE = 0;
		for (int i = 0; i < centraRes[centraRes.length - 1].length; i++) {
			double err = Math.abs(centraRes[centraRes.length - 1][i] - truth[truth.length - 1][i]);
			System.out.println(centraRes[centraRes.length - 1][i] + "\t" + truth[truth.length - 1][i]);
			absE += err / centraRes[centraRes.length - 1].length;
		}
		absErrorCSWA.add(absE);
	}

	public static void estimating(double _lp, double _lq, int wind, int latent) {
		List<ChainedFunction> lcf = new ArrayList<ChainedFunction>();
		for (CrowdSensor cs : cmap.values()) {
			lcf.add(cs.getRiskFunction(_lp, _lq, wind));
		}
		Matrix P, Q;
		int batch = 3;
		AveragedChainedRiskFunction arf = new AveragedChainedRiskFunction(lcf);

		ChainedMVariables cmv = LpMF.initiNMFPQ(new Matrix(getLatestWindow(cmap.get(0).collectedData, wind)), latent);
		ChainedMVariables res = GradientDescent.getMinimum(arf, cmv, 10e-4, 10e-2, 2000, GradientDescent.SGD);
		P = LpMF.getP(res);
		Q = LpMF.getQ(res);

		for (int i = 1; i < batch; i++) {
			lcf = new ArrayList<ChainedFunction>();
			for (CrowdSensor cs : cmap.values()) {
				lcf.add(cs.getRiskFunction(_lp, _lq, wind));
			}
			arf = new AveragedChainedRiskFunction(lcf);
			cmv = LpMF.initiNMFPQ(new Matrix(getLatestWindow(cmap.get(0).collectedData, wind)), latent);
			res = GradientDescent.getMinimum(arf, cmv, 10e-4, 10e-2, 2000, GradientDescent.SGD);
			P.plus(LpMF.getP(res));
			Q.plus(LpMF.getQ(res));
		}
		P.times(1.0 / batch);
		Q.times(1.0 / batch);
		// truncate(P, 0.8);
		// truncate(Q, 0.8);
		estimated = P.times(Q).getArray();
	}

	public static void centraEstimate(double _lp, double _lq, int wind, int latent) {
		List<ChainedFunction> lcf = new ArrayList<ChainedFunction>();
		for (CrowdSensor cs : cmap.values()) {
			lcf.add(cs.getRiskFunction(_lp, _lq, wind));
		}
		Matrix P, Q;
		AveragedChainedRiskFunction arf = new AveragedChainedRiskFunction(lcf);
		ChainedMVariables cmv = LpMF.initiNMFPQ(new Matrix(getLatestWindow(cmap.get(0).collectedData, wind)), latent);
		ChainedMVariables res = GradientDescent.getMinimum(arf, cmv, 10e-4, 10e-2, 2000, GradientDescent.GD);
		P = LpMF.getP(res);
		Q = LpMF.getQ(res);
		centraRes = P.times(Q).getArray();
	}

	public static void truncate(Matrix m, double rate) {
		List<Double> values = new ArrayList<Double>();
		for (int i = 0; i < m.getArray().length; i++) {
			for (int j = 0; j < m.getArray()[i].length; j++) {
				values.add(Math.abs(m.getArray()[i][j]));
			}
		}
		Collections.sort(values);
		// System.out.println(values.get(values.size()-1));
		double thr = values.get((int) (values.size() * (1.0 - rate)));
		for (int i = 0; i < m.getArray().length; i++) {
			for (int j = 0; j < m.getArray()[i].length; j++) {
				if (Math.abs(m.getArray()[i][j]) < thr)
					m.getArray()[i][j] = 0;
			}
		}
	}

	public static double[][] getLatestWindow(double[][] collectedData, int wind) {
		double[][] data = new double[wind][collectedData[0].length];
		int index = 0;
		for (int i = collectedData.length - wind; i < collectedData.length; i++) {
			for (int j = 0; j < collectedData[i].length; j++) {
				data[index][j] = collectedData[i][j];
			}
			index++;
		}
	//	System.out.println(data.length+"\t"+data[0].length);
		return data;
	}

	public static int[][] getCollectedCellsInLatestWindow(int[][] collectedData, int wind) {
		int[][] data = new int[wind][collectedData[0].length];
		int index = 0;
		for (int i = collectedData.length - wind; i < collectedData.length; i++) {
			for (int j = 0; j < collectedData[i].length; j++) {
				data[index][j] = collectedData[i][j];
			}
			index++;
		}
		return data;
	}

	public ChainedFunction getRiskFunction(double _lp, double _lq, int wind) {
		return LpMF.getNMFRiskFunction(new Matrix(getLatestWindow(this.collectedData, wind)), MFUtil.nmf, MFUtil.L1,
				getCollectedCellsInLatestWindow(this.nz, wind), _lp, _lq);
	}

}
