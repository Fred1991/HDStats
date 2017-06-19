package xiong.hdstats.opt.estimator;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import org.jtransforms.fft.DoubleFFT_2D;

import Jama.Matrix;
import xiong.hdstats.opt.ChainedFunction;
import xiong.hdstats.opt.ChainedMVariables;
import xiong.hdstats.opt.ChainedRiskFunction;
import xiong.hdstats.opt.GradientDescent;
import xiong.hdstats.opt.MatrixMVariable;
import xiong.hdstats.opt.MultiVariable;
import xiong.hdstats.opt.RiskFunction;

import ml.clustering.*;
import ml.options.L1NMFOptions;

public class L2NMF {

	public Matrix R;
	public Matrix P;
	public Matrix Q;
	public double lambdaP;
	public double lambdaQ;

	public L2NMF(Matrix _R, double _lp, double _lq) {
		this.R = _R;
		this.lambdaP = _lp;
		this.lambdaQ = _lq;
	}

	public void setP(Matrix _P) {
		this.P = project(_P);
	}

	public void setQ(Matrix _Q) {
		this.Q = project(_Q);
	}

	public Matrix func() {
		// TODO Auto-generated method stub
		double value = this.R.minus(P.times(Q)).normF() + this.lambdaP * P.normF() + this.lambdaQ * Q.normF();
		Matrix m = new Matrix(1, 1);
		m.set(0, 0, value);
		return m;
	}

	public MultiVariable gradientP() {
		// TODO Auto-generated method stub
		// System.out.println(R.getColumnDimension()+"\t"+R.getRowDimension());
		// System.out.println(P.getColumnDimension()+"\t"+P.getRowDimension());
		// R.minus(P.times(Q)).times(Q);
		// System.out.println("P gradient");
		Matrix gradientP = (R.minus(P.times(Q)).times(Q.transpose()).minus(P.times(lambdaP))).times(-1.0);
		// gradientP=gradientP.times(1.0/gradientP.norm2());
		// for(int i=0;i<gradientP.getRowDimension();i++){
		// for(int j=0;j<gradientP.getColumnDimension();j++){
		// System.out.println("P"+gradientP.get(i, j));
		// }
		// }
		// System.out.println(this.func().get(0,0));
		return new MatrixMVariable(gradientP);

	}

	public MultiVariable gradientQ() {
		// TODO Auto-generated method stub
		// System.out.println("Q gradient");
		Matrix gradientQ = (P.transpose().times(R.minus(P.times(Q))).minus(Q.times(lambdaQ))).times(-1.0);
		// gradientQ=gradientQ.times(1.0/gradientQ.norm2());

		// for(int i=0;i<gradientQ.getRowDimension();i++){
		// for(int j=0;j<gradientQ.getColumnDimension();j++){
		// System.out.println("Q"+gradientQ.get(i, j));
		// }
		// }
		// System.out.println(this.func().get(0,0));
		return new MatrixMVariable(gradientQ);
	}

	public static class NMFPRiskFunction implements RiskFunction {
		private L2NMF mf;

		public NMFPRiskFunction(L2NMF _mf) {
			this.mf = _mf;
		}

		@Override
		public Matrix func(MultiVariable input) {
			// TODO Auto-generated method stub
			ChainedMVariables PQ = (ChainedMVariables) input;
			this.mf.setP(((MatrixMVariable) PQ.get(0)).getMtx());
			this.mf.setQ(((MatrixMVariable) PQ.get(1)).getMtx());
			// System.out.println(this.mf.P);
			// System.out.println(this.mf.Q);
			return this.mf.func();
		}

		@Override
		public MultiVariable gradient(MultiVariable input) {
			// TODO Auto-generated method stub
			ChainedMVariables PQ = (ChainedMVariables) input;
			this.mf.setP(((MatrixMVariable) PQ.get(0)).getMtx());
			this.mf.setQ(((MatrixMVariable) PQ.get(1)).getMtx());
			// System.out.println(this.mf.P);
			// System.out.println(this.mf.Q);
			return this.mf.gradientP();
		}

	}

	public static class NMFQRiskFunction implements RiskFunction {

		private L2NMF mf;

		public NMFQRiskFunction(L2NMF _mf) {
			this.mf = _mf;
		}

		@Override
		public Matrix func(MultiVariable input) {
			// TODO Auto-generated method stub
			ChainedMVariables PQ = (ChainedMVariables) input;
			this.mf.setP(((MatrixMVariable) PQ.get(0)).getMtx());
			this.mf.setQ(((MatrixMVariable) PQ.get(1)).getMtx());
			// System.out.println(this.mf.P);
			// System.out.println(this.mf.Q);
			return this.mf.func();
		}

		@Override
		public MultiVariable gradient(MultiVariable input) {
			// TODO Auto-generated method stub
			ChainedMVariables PQ = (ChainedMVariables) input;
			this.mf.setP(((MatrixMVariable) PQ.get(0)).getMtx());
			this.mf.setQ(((MatrixMVariable) PQ.get(1)).getMtx());
			// System.out.println(this.mf.P);
			// System.out.println(this.mf.Q);
			return this.mf.gradientQ();
		}

	}

	public static ChainedFunction getNMFRiskFunction(Matrix R, double _lp, double _lq) {
		L2NMF mf = new L2NMF(R, _lp, _lq);
		List<RiskFunction> lrf = new ArrayList<RiskFunction>();
		lrf.add(new NMFPRiskFunction(mf));
		lrf.add(new NMFQRiskFunction(mf));
		ChainedRiskFunction crf = new ChainedRiskFunction(lrf);
		return crf;
	}

	public static ChainedMVariables initiNMFPQ(Matrix R, int latent) {
		List<MultiVariable> lms = new ArrayList<MultiVariable>();
		lms.add(new MatrixMVariable(Matrix.random(R.getRowDimension(), latent)));
		lms.add(new MatrixMVariable(Matrix.random(latent, R.getColumnDimension())));
		ChainedMVariables cmvs = new ChainedMVariables(lms);
		return cmvs;
	}

	public static Matrix project(Matrix p){
		for(int i=0;i<p.getRowDimension();i++){
			for(int j=0;j<p.getColumnDimension();j++){
				if(p.get(i, j)<0)
					p.set(i, j, 0);
			}
		}
		return p;
	}
	
	public static Matrix getP(ChainedMVariables cmv) {
		return project(((MatrixMVariable) cmv.get(0)).getMtx());
	}

	public static Matrix getQ(ChainedMVariables cmv) {
		return project(((MatrixMVariable) cmv.get(1)).getMtx());
	}

	public static double[][] convert(BufferedImage image) {
		int width = 128;
		int height = 128;
		double[][] result = new double[height][width];

		for (int row = 0; row < height; row++) {
			for (int col = 0; col < width; col++) {
				result[row][col] = image.getRGB(col, row);
			}
		}

		return result;
	}

	public static void main(String[] args) throws IOException {
		BufferedImage hugeImage = ImageIO.read(new File("C:\\Users\\jbn42\\Desktop\\leyewang.png"));
		double[][] original = convert(hugeImage);
		DoubleFFT_2D fft=new DoubleFFT_2D(original.length,original[0].length);
		fft.realForward(original);

		Matrix orginal = new Matrix(original);
		Matrix noise1 = Matrix.random(orginal.getRowDimension(), orginal.getColumnDimension());
		Matrix noise2 = (Matrix.random(orginal.getRowDimension(), orginal.getColumnDimension()));
		Matrix noise = noise1.minus(noise2);
		noise = noise.times(0.01);
		Matrix toM = orginal.plus(noise);
		System.out.println(noise.normF());
		BufferedImage bufferedImage = new BufferedImage(original.length, original[0].length,
				BufferedImage.TYPE_INT_RGB);

		// Set each pixel of the BufferedImage to the color from the Color[][].
		double[][] noiseAdd = toM.getArrayCopy();
		fft.realInverse(noiseAdd, true);
		for (int x = 0; x < noiseAdd.length; x++) {
			for (int y = 0; y < noiseAdd[x].length; y++) {
				bufferedImage.setRGB(x, y, ((int) noiseAdd[x][y]));
			}
		}

		ImageIO.write(bufferedImage, "PNG", new File("C:\\Users\\jbn42\\Desktop\\leyewang2.png"));

		L1NMFOptions L1NMFOptions = new L1NMFOptions();
		L1NMFOptions.nClus = 50;
		L1NMFOptions.gamma = 1 * 0.0001;
		L1NMFOptions.mu = 1 * 0.1;
		L1NMFOptions.maxIter = 1000;
		L1NMFOptions.verbose = true;
		L1NMFOptions.calc_OV = !true;
		L1NMFOptions.epsilon = 1e-5;
		Clustering L1NMF = new L1NMF(L1NMFOptions);
		L1NMF.feedData(toM.getArrayCopy());
		L1NMF.clustering();
		la.matrix.Matrix L= L1NMF.getCenters();
		la.matrix.Matrix R = L1NMF.getIndicatorMatrix();
		L=L.transpose();
		R=R.transpose();
		System.out.println(L.getRowDimension()+"\t"+L.getColumnDimension());
		System.out.println(R.getRowDimension()+"\t"+R.getColumnDimension());
		Matrix G = new Matrix(L.getData()).times(new Matrix(R.getData()));
		System.out.println(G.getRowDimension()+"\t"+G.getColumnDimension());

		double[][] recovered =G.getArrayCopy();
		fft.realInverse(recovered, true);
		System.out.println("nmf\t" + G.minus(orginal).norm1() / orginal.norm1());

		BufferedImage bufferedImage2 = new BufferedImage(original.length, original[0].length,
				BufferedImage.TYPE_INT_RGB);

		// Set each pixel of the BufferedImage to the color from the Color[][].

		for (int x = 0; x < noiseAdd.length; x++) {
			for (int y = 0; y < noiseAdd[x].length; y++) {
				bufferedImage2.setRGB(x, y, ((int) recovered[x][y]));
			}
		}

		ImageIO.write(bufferedImage2, "PNG", new File("C:\\Users\\jbn42\\Desktop\\leyewang4.png"));

		
		// for (int i = 2; i <50; i += 1) {
		int i = 50;
		ChainedFunction cf = L2NMF.getNMFRiskFunction(toM, 0.0001, 0.0001);
		ChainedMVariables cmv = L2NMF.initiNMFPQ(toM, i);
		ChainedMVariables res = GradientDescent.getMinimum(cf, cmv, 10e-16, 10e-4, 1000, GradientDescent.GD);
		Matrix P = getP(res);
		Matrix Q = getQ(res);
		System.out.println(i + "\t" + P.times(Q).minus(orginal).norm1() / orginal.norm1());
		// }
		recovered = P.times(Q).getArrayCopy();
		BufferedImage bufferedImage3 = new BufferedImage(original.length, original[0].length,
				BufferedImage.TYPE_INT_RGB);

		// Set each pixel of the BufferedImage to the color from the Color[][].
		fft.realInverse(recovered, true);

		for (int x = 0; x < noiseAdd.length; x++) {
			for (int y = 0; y < noiseAdd[x].length; y++) {
				bufferedImage3.setRGB(x, y, ((int) recovered[x][y]));
			}
		}

		ImageIO.write(bufferedImage3, "PNG", new File("C:\\Users\\jbn42\\Desktop\\leyewang3.png"));

	}

}