package xiong.hdstats.opt;


public interface MultiVariable {
	public void updatedByGradient(MultiVariable gradient, double eta);
	public MultiVariable clone();
	public MultiVariable plus(MultiVariable m);
	public MultiVariable times(double d);
	public double scalar();
}
