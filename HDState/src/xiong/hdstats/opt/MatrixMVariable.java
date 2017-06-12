package xiong.hdstats.opt;

import Jama.Matrix;

public class MatrixMVariable implements MultiVariable{
	private Matrix mtx;
	
	public MatrixMVariable(Matrix _m){
		this.mtx=_m;
	}
	
	public Matrix getMtx(){
		return this.mtx;
	}

	@Override
	public void updatedByGradient(MultiVariable gradient, double eta) {
		// TODO Auto-generated method stub
		Matrix gmtx = ((MatrixMVariable)gradient).mtx;
		this.mtx=this.mtx.plus(gmtx.times(-1.0*eta));
	}

	@Override
	public MultiVariable clone() {
		// TODO Auto-generated method stub
		return new MatrixMVariable(this.mtx.copy());
	}

	@Override
	public MultiVariable plus(MultiVariable m) {
		// TODO Auto-generated method stub
		return new MatrixMVariable(this.mtx.plus(((MatrixMVariable)m).mtx));
	}

	@Override
	public MultiVariable times(double d) {
		// TODO Auto-generated method stub
		return new MatrixMVariable(this.mtx.times(d));
	}

	@Override
	public double scalar() {
		// TODO Auto-generated method stub
		return mtx.normF();
	}

}
