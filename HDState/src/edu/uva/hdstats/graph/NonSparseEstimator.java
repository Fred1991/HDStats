package edu.uva.hdstats.graph;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.UUID;

import Jama.Matrix;
import edu.uva.hdstats.MLEstimator;
import edu.uva.hdstats.da.PseudoInverse;

public class NonSparseEstimator extends MLEstimator {
	private double _lambda = 0.01;

	public NonSparseEstimator() {
	}

	public NonSparseEstimator(double lambda) {
		this._lambda = lambda;
	}

	@Override
	public double[][] covariance(double[][] samples) {
		return _deSparsifiedGlassoGetCovarianceMatrix(super.covariance(samples));
		// covarianceApprox(covar_inner);
		// Matrix m = new Matrix(precision_matrix);
		// return m.inverse().getArray();
		// return m.getArray();
	}

	@Override
	public void covarianceApprox(double[][] covar_inner) {
		double[][] sparsifiedGlassoCov = _deSparsifiedGlassoGetCovarianceMatrix(covar_inner);
		for (int i = 0; i < covar_inner.length; i++) {
			for (int j = 0; j < covar_inner[i].length; j++) {
				covar_inner[i][j] = sparsifiedGlassoCov[i][j];
			}
		}
	}

	public double[][] _deSparsifiedGlassoGetCovarianceMatrix(double[][] covx) {
		String id = UUID.randomUUID().toString();
		double[][] inverseCovarianceMatrix = new double[covx.length][covx.length];
		/// System.out.println("data length:"+data[0].length);
		try {
			PrintWriter writer = new PrintWriter(new FileWriter(R_src_Path+"R_non_sparse_tmp" + id + ".data"));
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
			PrintWriter writer = new PrintWriter(new FileWriter(R_src_Path+"R_non_sparse_tmp" + id + ".R"));
			writer.println("library(glasso)");
			writer.println("library(Matrix)");
			writer.println("library(MASS)");
			writer.println("library(matrixcalc)");
			writer.println("R_dataset = read.csv(\""+R_src_Path+"R_non_sparse_tmp" + id + ".data\", header=FALSE)");
			// writer.println("R_dataset");
			writer.println("R_covarianceMatrix = as.matrix(R_dataset)");
			// writer.println("R_covarianceMatrix <- nearPD(R_covarianceMatrix,
			// corr=FALSE, keepDiag=TRUE, do2eigen=TRUE, doSym=TRUE,
			// doDykstra=TRUE)");
			// writer.println("R_covarianceMatrix<-as.matrix(R_covarianceMatrix$mat)");

			// writer.println("R_covarianceMatrix[300,600]");
			writer.println(
					"r_non_sparse = glasso(R_covarianceMatrix, rho=" + this._lambda + ", penalize.diagonal=TRUE)");
			writer.println("r_non_sparse<-as.matrix(r_non_sparse$wi)");
			writer.println(
					"Zettahat<-(r_non_sparse + r_non_sparse)-(r_non_sparse %*% R_covarianceMatrix %*% r_non_sparse)");

			// writer.println("if(is.singular.matrix(Zettahat)){");
			// writer.println("Sigmahat<-ginv(Zettahat)");
			// writer.println("}else{");
			writer.println("Sigmahat<-solve(Zettahat)");
			// writer.println("}");

			writer.println("write(t(Sigmahat), file=\""+R_src_Path+"r_non_sparse_wi_tmp" + id + ".txt\", "
					+ "ncolumns=dim(Sigmahat)[[2]], sep=\",\")");
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		// execute "Rscript R_non_sparse_tmp.R"
		try {
			Process p = Runtime.getRuntime().exec("/usr/local/bin/Rscript "+R_src_Path+"R_non_sparse_tmp" + id + ".R");

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
			BufferedReader inputReader = new BufferedReader(new FileReader(R_src_Path+"r_non_sparse_wi_tmp" + id + ".txt"));
			for (int i = 0; i < inverseCovarianceMatrix.length; i++) {
				String line = inputReader.readLine();
				String[] lns = line.split(",");
				// System.out.println("r size:"+lns.length);
				// System.out.println("r output:"+line);
				// StringTokenizer t = new StringTokenizer(line, "\t");
				for (int j = 0; j < inverseCovarianceMatrix[i].length; j++) {
					try {
						inverseCovarianceMatrix[i][j] = Double.parseDouble(lns[j]);
					} catch (Exception e) {
						if (lns[j].toLowerCase().startsWith("inf")) {
							inverseCovarianceMatrix[i][j] = Double.POSITIVE_INFINITY;
						} else if (lns[j].toLowerCase().startsWith("-inf")) {
							inverseCovarianceMatrix[i][j] = Double.NEGATIVE_INFINITY;
						} else {
							inverseCovarianceMatrix[i][j] = 0;
						}
					}
				}
			}
			inputReader.close();
			
			new File(R_src_Path+"R_non_sparse_tmp"+id+".txt").delete();
			new File(R_src_Path+"R_non_sparse_tmp"+id+".data").delete();
			new File(R_src_Path+"r_non_sparse_wi_tmp"+id+".txt").delete();

		} catch (IOException e) {
			e.printStackTrace();
		}
		return inverseCovarianceMatrix;
		// try {
		// return new Matrix(inverseCovarianceMatrix).inverse().getArray();
		// } catch (Exception e) {
		// e.printStackTrace();
		// return PseudoInverse.inverse(new
		// Matrix(inverseCovarianceMatrix)).getArray();
		// }
	}

	public double[][] _deSparsifiedGlassoPrecisionMatrix(double[][] covx) {
		String id = UUID.randomUUID().toString();
		double[][] inverseCovarianceMatrix = new double[covx.length][covx.length];
		/// System.out.println("data length:"+data[0].length);
		try {
			PrintWriter writer = new PrintWriter(new FileWriter(R_src_Path+"R_non_sparse_tmp" + id + ".data"));
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
			PrintWriter writer = new PrintWriter(new FileWriter(R_src_Path+"R_non_sparse_tmp" + id + ".R"));
			writer.println("library(glasso)");
			writer.println("library(Matrix)");
			writer.println("library(MASS)");
			writer.println("library(matrixcalc)");
			writer.println("R_dataset = read.csv(\""+R_src_Path+"R_non_sparse_tmp" + id + ".data\", header=FALSE)");
			// writer.println("R_dataset");
			writer.println("R_covarianceMatrix = as.matrix(R_dataset)");
		//	writer.println(
		//			"R_covarianceMatrix <- nearPD(R_covarianceMatrix, corr=FALSE, keepDiag=FALSE, do2eigen=TRUE, doSym=TRUE, doDykstra=TRUE)");
		//	writer.println("R_covarianceMatrix<-as.matrix(R_covarianceMatrix$mat)");
			// writer.println("R_covarianceMatrix[300,600]");
			writer.println("r_glasso = glasso(R_covarianceMatrix, rho=" + this._lambda + ", penalize.diagonal=TRUE)");
			writer.println("r_glasso<-as.matrix(r_glasso$wi)");
			writer.println("Zettahat<-(r_glasso + r_glasso)-(r_glasso %*% R_covarianceMatrix %*% r_glasso)");

			writer.println("write(t(Zettahat), file=\""+R_src_Path+"r_non_sparse_wi_tmp" + id + ".txt\", "
					+ "ncolumns=dim(Zettahat)[[2]], sep=\",\")");
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		// execute "Rscript R_non_sparse_tmp.R"
		try {
			Process p = Runtime.getRuntime().exec("/usr/local/bin/Rscript "+R_src_Path+"R_non_sparse_tmp" + id + ".R");

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
			System.exit(-1);
			e.printStackTrace();
		}

		try {
			BufferedReader inputReader = new BufferedReader(new FileReader(R_src_Path+"r_non_sparse_wi_tmp" + id + ".txt"));
			for (int i = 0; i < inverseCovarianceMatrix.length; i++) {
				String line = inputReader.readLine();
				String[] lns = line.split(",");
				// System.out.println("r size:"+lns.length);
				// System.out.println("r output:"+line);
				// StringTokenizer t = new StringTokenizer(line, "\t");
				for (int j = 0; j < inverseCovarianceMatrix[i].length; j++) {
					try {
						inverseCovarianceMatrix[i][j] = Double.parseDouble(lns[j]);
					} catch (Exception e) {
						if (lns[j].toLowerCase().startsWith("inf")) {
							inverseCovarianceMatrix[i][j] = Double.POSITIVE_INFINITY;
						} else if (lns[j].toLowerCase().startsWith("-inf")) {
							inverseCovarianceMatrix[i][j] = Double.NEGATIVE_INFINITY;
						} else {
							inverseCovarianceMatrix[i][j] = Double.NaN;
						}
					}
				}
			}
			inputReader.close();
			
			new File(R_src_Path+"R_non_sparse_tmp"+id+".txt").delete();
			new File(R_src_Path+"R_non_sparse_tmp"+id+".data").delete();
			new File(R_src_Path+"r_non_sparse_wi_tmp"+id+".txt").delete();

		} catch (IOException e) {
			e.printStackTrace();
		}

		return inverseCovarianceMatrix;
	}

}
