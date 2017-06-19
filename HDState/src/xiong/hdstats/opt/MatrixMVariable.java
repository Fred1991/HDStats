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
	//	System.out.println("here");
		Matrix gmtx = ((MatrixMVariable)gradient).mtx;
		Matrix res=this.mtx.plus(gmtx.times(-1.0*eta));
		for(int i=0;i<mtx.getRowDimension();i++){
			for(int j=0;j<mtx.getColumnDimension();j++){
				mtx.set(i, j, res.get(i, j));
			}
		}
	}

	@Override
	public MultiVariable clone() {
		// TODO Auto-generated method stub
	//	System.out.println("clone");
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
