package xiong.hdstats.opt.var;

import java.util.ArrayList;
import java.util.List;

import xiong.hdstats.opt.MultiVariable;

public class StructuredMVariable implements MultiVariable {

	public List<MultiVariable> mvs = new ArrayList<MultiVariable>();

	public StructuredMVariable(List<MultiVariable> _m) {
		for (MultiVariable m : _m)
			mvs.add(m.clone());
	}

	public List<MultiVariable> getMVS() {
		return this.mvs;
	}

	@Override
	public void updatedByGradient(MultiVariable gradient, double eta) {
		// TODO Auto-generated method stub
		StructuredMVariable gls = (StructuredMVariable) gradient;
		for (int i = 0; i < mvs.size(); i++) {
			mvs.get(i).updatedByGradient(gls.mvs.get(i), eta);
		}
	}

	@Override
	public MultiVariable clone() {
		// TODO Auto-generated method stub
		return new StructuredMVariable(this.getMVS());
	}

	@Override
	public MultiVariable plus(MultiVariable m) {
		// TODO Auto-generated method stub
		StructuredMVariable mamv=(StructuredMVariable) this.clone();
		mamv.updatedByGradient(m, -1);
		return mamv;
	}

	@Override
	public MultiVariable times(double d) {
		// TODO Auto-generated method stub
		StructuredMVariable mamv=(StructuredMVariable) this.clone();
		for(MultiVariable m:mamv.mvs)
			m.times(d);
		return mamv;
	}

	@Override
	public double scalar() {
		// TODO Auto-generated method stub
		double sum=0;
		for(MultiVariable mv:mvs)
			sum+=mv.scalar();
		return sum;
	}

}
