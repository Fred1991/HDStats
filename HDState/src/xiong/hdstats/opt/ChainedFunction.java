package xiong.hdstats.opt;

public interface ChainedFunction extends RiskFunction{
	public RiskFunction getCurrent();
	public boolean innerLoop();
	public void toNext();
}
