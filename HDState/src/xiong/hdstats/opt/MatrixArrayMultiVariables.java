package xiong.hdstats.opt;

import java.util.ArrayList;
import java.util.List;

import Jama.Matrix;

public class MatrixArrayMultiVariables implements MultiVariable {

	public List<MatrixMVariable> mvs = new ArrayList<MatrixMVariable>();

	public MatrixArrayMultiVariables(List<Matrix> _m) {
		for (Matrix m : _m)
			mvs.add(new MatrixMVariable(m));
	}

	public List<Matrix> getMtxs() {
		List<Matrix> mtxs = new ArrayList<Matrix>();
		for (MatrixMVariable mv : mvs)
			mtxs.add(mv.getMtx());
		return mtxs;
	}

	@Override
	public void updatedByGradient(MultiVariable gradient, double eta) {
		// TODO Auto-generated method stub
		MatrixArrayMultiVariables gls = (MatrixArrayMultiVariables) gradient;
		for (int i = 0; i < mvs.size(); i++) {
			mvs.get(i).updatedByGradient(gls.mvs.get(i), eta);
		}
	}

	@Override
	public MultiVariable clone() {
		// TODO Auto-generated method stub
		return new MatrixArrayMultiVariables(this.getMtxs());
	}

	@Override
	public MultiVariable plus(MultiVariable m) {
		// TODO Auto-generated method stub
		MatrixArrayMultiVariables mamv=(MatrixArrayMultiVariables) this.clone();
		mamv.updatedByGradient(m, -1);
		return mamv;
	}

	@Override
	public MultiVariable times(double d) {
		// TODO Auto-generated method stub
		MatrixArrayMultiVariables mamv=(MatrixArrayMultiVariables) this.clone();
		for(MatrixMVariable m:mamv.mvs)
			m.times(d);
		return mamv;
	}

	@Override
	public double scalar() {
		// TODO Auto-generated method stub
		double sum=0;
		for(MatrixMVariable mv:mvs)
			sum+=mv.scalar();
		return sum;
	}

}
