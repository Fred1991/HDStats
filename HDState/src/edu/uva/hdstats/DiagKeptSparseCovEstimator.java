package edu.uva.hdstats;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.UUID;

public class DiagKeptSparseCovEstimator extends MLEstimator {

	public int _iter;
	public double _lambda;

	public DiagKeptSparseCovEstimator(double lambda, int iter) {
		this._lambda = lambda;
		this._iter = iter;
	}

	@Override
	public double[][] covariance(double[][] samples) {
		double[][] covar_inner = super.covariance(samples);
		covarianceApprox(covar_inner);
		return covar_inner;
	}

	
	@Override
	public void covarianceApprox(double[][] covar_inner){
		DiagKeptLassoEstimator le = new DiagKeptLassoEstimator(this._lambda);
		for (int i = 0; i < _iter; i++) {
			le.covarianceApprox(covar_inner);
		//	NearPD npd = new NearPD();
		//	npd.calcNearPD(new Jama.Matrix(covar_inner));
		//	double[][] covarx = npd.getX().getArrayCopy();
			double[][] covarx=nearestPSD(covar_inner);
			for(int k=0;k<covarx.length;k++){
				for(int j=0;j<covarx[k].length;j++){
					covar_inner[k][j]=covarx[k][j];
				}
			}
		}
	}
	
	
	private double[][] nearestPSD(double[][] covx){
		String id=UUID.randomUUID().toString();
		double[][] psdMatrix = new double[covx.length][covx.length];
		/// System.out.println("data length:"+data[0].length);
		try {
			PrintWriter writer = new PrintWriter(new FileWriter(R_src_Path+"R_sparse_tmp"+id+".data"));
			// writer.print("variable_0");
			// for(int i = 1; i < covx[0].length; i++)
			// writer.print(",variable_"+i);
			System.out.println(covx[0].length + " x " + covx.length);
			System.out.println("cols," + covx[0].length);

			for (int i = 0; i < covx.length; i++) {
				writer.print(covx[i][0]);
				for (int j = 1; j < covx[i].length; j++) {
					writer.print("," + covx[i][j]);
				}
				writer.print("\n");
				writer.println();
			}
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		try {
			PrintWriter writer = new PrintWriter(new FileWriter(R_src_Path+"R_sparse_tmp"+id+".R"));
			writer.println("library(Matrix)");
			writer.println("R_dataset = read.csv(\""+R_src_Path+"R_sparse_tmp"+id+".data\", header=FALSE)");
			// writer.println("R_dataset");
			writer.println("R_covarianceMatrix = as.matrix(R_dataset)");
			// writer.println("R_covarianceMatrix[300,600]");
			writer.println("res <- nearPD(R_covarianceMatrix, corr=FALSE, keepDiag=TRUE, do2eigen=TRUE, doSym=TRUE, doDykstra=TRUE)");
		  //   writer.println("res$mat");
			writer.println("write(t(as.matrix(res$mat)), file=\""+R_src_Path+"R_sparse_wi_tmp"+id+".txt\", "
					+ "ncolumns=dim(res$mat)[[2]], sep=\",\")");

			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		// execute "Rscript R_sparse_tmp.R"
		try {
			System.out.println("to execute Rscript "+R_src_Path+"R_sparse_tmp"+id+".R");
			Process p = Runtime.getRuntime().exec("/usr/local/bin/Rscript "+R_src_Path+"R_sparse_tmp"+id+".R");

			String s;

			// read the output from the command
			BufferedReader stdInput = new BufferedReader(new InputStreamReader(p.getInputStream()));
			// System.out.println("********************************************************");
			// System.out.println("Here is the standard output of the
			// command:\n");
			while ((s = stdInput.readLine()) != null) {
				System.out.println(s);
			}
			// System.out.println("********************************************************");
			// System.out.println();
			stdInput.close();

			// read any errors from the command
			BufferedReader stdError = new BufferedReader(new InputStreamReader(p.getErrorStream()));
			// System.out.println("********************************************************");
			// System.out.println("Here is the standard error of the command (if
			// any):\n");
			while ((s = stdError.readLine()) != null) {
				System.out.println(s);
			}
			// System.out.println("********************************************************");
			// System.out.println();
			stdError.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		try {
			BufferedReader inputReader = new BufferedReader(new FileReader(R_src_Path+"R_sparse_wi_tmp"+id+".txt"));
			for (int i = 0; i < psdMatrix.length; i++) {
				String line = inputReader.readLine();
				String[] lns = line.split(",");
				// System.out.println("r size:"+lns.length);
				// System.out.println("r output:"+line);
				// StringTokenizer t = new StringTokenizer(line, "\t");
				for (int j = 0; j < psdMatrix[i].length; j++) {
					psdMatrix[i][j] = Double.parseDouble(lns[j]);
				}
			}
			inputReader.close();
			new File(R_src_Path+"R_sparse_tmp"+id+".txt").delete();
			new File(R_src_Path+"R_sparse_tmp"+id+".data").delete();
			new File(R_src_Path+"R_sparse_wi_tmp"+id+".txt").delete();

		} catch (IOException e) {
			e.printStackTrace();
		}

		return psdMatrix;
	}
	

}
