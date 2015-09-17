package edu.uva.libopt.numeric;

public interface NumericFunction {
	
	public double func(double[] X);
	public double[] gradient(double[] X);
}
