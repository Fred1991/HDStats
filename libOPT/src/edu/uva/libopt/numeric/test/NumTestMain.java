package edu.uva.libopt.numeric.test;

import edu.uva.libopt.numeric.*;
import edu.uva.libopt.numeric.optimizer.*;

public class NumTestMain {

	public static class SBFunction implements NumericFunction{
		@Override
		public double func(double[] X) {
			// TODO Auto-generated method stub
		 		return X[0]* X[0]+3* X[0]+
		 			   X[1]* X[1]+6* X[1]+
		 			   X[2]* X[2]+9* X[2]+
		 			   X[3]* X[3]+12* X[3]+
		 			   X[4]* X[4]+15* X[4]+
		 			   X[5]* X[5]+18* X[5]+
		 			   X[6]* X[6]+21* X[6]+
		 			   X[7]* X[7]+24* X[7]+
		 			   X[8]* X[8]+27* X[8]+
		 			   X[9]* X[9]+30* X[9]+
		 			   Math.abs(X[0])+
		 			   Math.abs(X[1])+
		 			   Math.abs(X[2])+
		 			   Math.abs(X[3])+
		 			   Math.abs(X[4])+
		 			   Math.abs(X[5])+
		 			   Math.abs(X[6])+
		 			   Math.abs(X[7])+
		 			   Math.abs(X[8])+
		 			   Math.abs(X[9]);
		}

		@Override
		public double[] gradient(double[] X) {
			// TODO Auto-generated method stub
			return Utils.getGradient(this, X, 0.000001);
		}
		
	}
	
	
	public static void main(String[] args){
		NumericOptimizer nopt=new GradientDescentOptimizer();
		double[] x=new double[10];
		double minf=nopt.getMinimum(x, 0.0000000001, 0.00000000001, new SBFunction());
		for(int index=0;index<x.length;index++)
			System.out.println("x_"+index+":"+x[index]);
		System.out.println("Y:"+minf);
	}
	
}
