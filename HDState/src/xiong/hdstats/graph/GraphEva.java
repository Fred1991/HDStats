package xiong.hdstats.graph;

import java.io.PrintStream;

public class GraphEva {
	public int nodes;
	public int edges;
	public int falsediscovery;
	public int recovery;
	public int tp,tn,fp,fn;
	public double precision;
	public double recall;
	public double F1score;
	public double sensitivity;
	public double specificity;
	public int support;
	
	public GraphEva(int[][] truth, int[][] estimated){
		for(int i=0;i<truth.length;i++){
			for(int j=0;j<truth.length;j++){
				if(truth[i][j]!=0)
					support++;
				
				if(estimated[i][j]!=0)
					edges++;
				
				if(truth[i][j]!=0&&estimated[i][j]!=0){
					recovery++;
					this.tp++;
				}else if(truth[i][j]==0&&estimated[i][j]!=0){
					falsediscovery++;
					this.fp++;
				}else if(truth[i][j]==0&&estimated[i][j]==0){
					this.tn++;
				}else{
					this.fn++;
				}
			
			}
		}
//		this.precision=this.tp/(this.tp+this.fp);
//		this.recall=this.tp/(this.tp+this.fn);
//		this.F1score=2*this.precision*this.recall/(this.precision+this.recall);
//		this.sensitivity=this.recall;
//		this.specificity=this.tn/(this.tn+this.fp);
	}
	
	public void print(String name, PrintStream ps){
		ps.println(name
				+"\t"+this.support
				+"\t"+this.edges
				+"\t"+this.tp
				+"\t"+this.tn
				+"\t"+this.fp
				+"\t"+this.fn);
	}
}
