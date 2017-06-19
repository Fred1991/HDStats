package xiong.hdstats.opt;

import java.util.ArrayList;
import java.util.List;

public class ChainedMVariables implements MultiVariable {

	private List<MultiVariable> mvs = new ArrayList<MultiVariable>();
	private int index = 0;

	public MultiVariable get(int in) {
		return this.mvs.get(in);
	}

	public ChainedMVariables(List<MultiVariable> mv) {
		for (MultiVariable m : mv)
			mvs.add(m);
	}

	public void toNext() {
		index++;
		index = index % mvs.size();
	}

	public MultiVariable getCurrent() {
		return this.mvs.get(index);
	}

	@Override
	public void updatedByGradient(MultiVariable gradient, double eta) {
		// TODO Auto-generated method stub
		mvs.get(index).updatedByGradient(gradient, eta);
	}

	@Override
	public ChainedMVariables clone() {
		// TODO Auto-generated method stub
		List<MultiVariable> lmvs = new ArrayList<MultiVariable>();
		for (MultiVariable mv : mvs)
			lmvs.add(mv.clone());
		ChainedMVariables cmvs = new ChainedMVariables(lmvs);
		return cmvs;
	}

	@Override
	public MultiVariable plus(MultiVariable m) {
		// TODO Auto-generated method stub
		return mvs.get(index).plus(m);
	}

	@Override
	public MultiVariable times(double d) {
		// TODO Auto-generated method stub
		return mvs.get(index).times(d);
	}

	@Override
	public double scalar() {
		// TODO Auto-generated method stub
		return mvs.get(index).scalar();
	}

}
