package xiong.hdstats.gaussian.online;

public interface OnlineGraphEstimator {
	public void update(int index, double[] newdata);
	public void init(double[][] samples);
	public double[][] getGraph();
}
