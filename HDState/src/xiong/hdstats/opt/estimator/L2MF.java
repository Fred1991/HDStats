package xiong.hdstats.opt.estimator;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import Jama.Matrix;
import xiong.hdstats.opt.ChainedFunction;
import xiong.hdstats.opt.ChainedRiskFunction;
import xiong.hdstats.opt.GradientDescent;
import xiong.hdstats.opt.var.MatrixMVariable;
import xiong.hdstats.opt.MultiVariable;
import xiong.hdstats.opt.RiskFunction;
import xiong.hdstats.opt.var.ChainedMVariables;

public class L2MF {

	public Matrix R;
	public Matrix P;
	public Matrix Q;
	public double lambdaP;
	public double lambdaQ;

	public L2MF(Matrix _R, double _lp, double _lq) {
		this.R = _R;
		this.lambdaP = _lp;
		this.lambdaQ = _lq;
	}

	public void setP(Matrix _P) {
		this.P = _P;
	}

	public void setQ(Matrix _Q) {
		this.Q = _Q;
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

	public static class PRiskFunction implements RiskFunction {
		private L2MF mf;

		public PRiskFunction(L2MF _mf) {
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

	public static class QRiskFunction implements RiskFunction {

		private L2MF mf;

		public QRiskFunction(L2MF _mf) {
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

	public static ChainedFunction getRiskFunction(Matrix R, double _lp, double _lq) {
		L2MF mf = new L2MF(R, _lp, _lq);
		List<RiskFunction> lrf = new ArrayList<RiskFunction>();
		lrf.add(new PRiskFunction(mf));
		lrf.add(new QRiskFunction(mf));
		ChainedRiskFunction crf = new ChainedRiskFunction(lrf);
		return crf;
	}

	public static ChainedMVariables initiPQ(Matrix R, int latent) {
		List<MultiVariable> lms = new ArrayList<MultiVariable>();
		lms.add(new MatrixMVariable(Matrix.random(R.getRowDimension(), latent)));
		lms.add(new MatrixMVariable(Matrix.random(latent, R.getColumnDimension())));
		ChainedMVariables cmvs = new ChainedMVariables(lms);
		return cmvs;
	}

	public static Matrix getP(ChainedMVariables cmv) {
		return ((MatrixMVariable) cmv.get(0)).getMtx();
	}

	public static Matrix getQ(ChainedMVariables cmv) {
		return ((MatrixMVariable) cmv.get(1)).getMtx();
	}

	public static double[][] convert(BufferedImage image) {
		int width = image.getWidth();
		int height = image.getHeight();
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
		double[][] diff = new double[original.length][original[0].length];
		for (int x = 0; x < original.length; x++) {
			for (int y = 0; y < original[x].length; y++) {
				diff[x][y] = original[x][y] - original[(x + 1) % original.length][(y + 1) % original[x].length];
			}
		}

		Matrix orginal = new Matrix(diff);
		Matrix noise1 = Matrix.random(orginal.getRowDimension(), orginal.getColumnDimension());
		Matrix noise2 = (Matrix.random(orginal.getRowDimension(), orginal.getColumnDimension()));
		Matrix noise = noise1.minus(noise2);
		noise = noise.times(0);
		Matrix toM = orginal.plus(noise);
		System.out.println(noise.normF());
		BufferedImage bufferedImage = new BufferedImage(original.length, original[0].length,
				BufferedImage.TYPE_INT_RGB);

		// Set each pixel of the BufferedImage to the color from the Color[][].
		double[][] noiseAdd = toM.getArray();
		for (int x = 0; x < noiseAdd.length; x++) {
			for (int y = 0; y < noiseAdd[x].length; y++) {
				bufferedImage.setRGB(x, y, ((int) noiseAdd[x][y]));
			}
		}

		ImageIO.write(bufferedImage, "PNG", new File("C:\\Users\\jbn42\\Desktop\\leyewang2.png"));

		// for (int i = 2; i <50; i += 1) {
		int i = 30;
		ChainedFunction cf = getRiskFunction(toM, 0, 0);
		ChainedMVariables cmv = initiPQ(toM, i);
		ChainedMVariables res = GradientDescent.getMinimum(cf, cmv, 10e-11, 10e-4, 1000, GradientDescent.GD);
		Matrix P = getP(res);
		Matrix Q = getQ(res);
		System.out.println(i + "\t" + P.times(Q).minus(orginal).norm1() / orginal.norm1());
		System.out.println(i + "loss\t" + toM.minus(orginal).norm1() / orginal.norm1());
		// }
		double[][] recovered = P.times(Q).getArray();
		BufferedImage bufferedImage2 = new BufferedImage(original.length, original[0].length,
				BufferedImage.TYPE_INT_RGB);

		// Set each pixel of the BufferedImage to the color from the Color[][].
		for (int x = 0; x < noiseAdd.length; x++) {
			for (int y = 0; y < noiseAdd[x].length; y++) {
				bufferedImage2.setRGB(x, y, ((int) recovered[x][y]));
			}
		}

		ImageIO.write(bufferedImage2, "PNG", new File("C:\\Users\\jbn42\\Desktop\\leyewang3.png"));

	}

}
