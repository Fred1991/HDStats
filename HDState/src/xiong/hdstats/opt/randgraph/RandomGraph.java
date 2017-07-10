package xiong.hdstats.opt.randgraph;

//This is a java program to generate a random graph using random edge generation
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class RandomGraph {
	private Map<Integer, List<Integer>> adjacencyList;
	private Random seed;
	private int numEdges = 0;
	private int[] maxDegree;
	private int current;

	public RandomGraph(int v) {
		adjacencyList = new HashMap<Integer, List<Integer>>();
		this.maxDegree = new int[v];
		for (int i = 1; i <= v; i++) {
			adjacencyList.put(i, new LinkedList<Integer>());
			this.setMaxDegree(i, Integer.MAX_VALUE);
		}
		seed = new Random(); 
	}

	public int nextInt() {
		return this.getNextNodeByRandom(current) - 1;  //to index
	}

	public int randStart() {
		this.current = seed.nextInt(adjacencyList.size()) + 1;
		return current;
	}

	public void setMaxDegree(int v, int degree) {
		this.maxDegree[v - 1] = degree;
	}

	public int getNumEdges() {
		return this.numEdges;
	}

	public void setEdge(int from, int to) {
		if (to == from)
			return;
		if (to > adjacencyList.size() || from > adjacencyList.size()) {
			System.out.println("The vertices does not exists");
			return;
		}
		List<Integer> sls = adjacencyList.get(to);
		List<Integer> dls = adjacencyList.get(from);

		if (sls.size() < this.maxDegree[from - 1]) {
			if (!sls.contains(from)) {
				sls.add(from);
				this.numEdges++;
			}
			if (!dls.contains(to))
				dls.add(to);
		}
	}

	public List<Integer> getEdge(int to) {
		if (to > adjacencyList.size()) {
			System.out.println("The vertices does not exists");
			return null;
		}
		return adjacencyList.get(to);
	}

	public int getNextNodeByRandom(int current) {
		List<Integer> cand = this.getEdge(current);
		return cand.get(seed.nextInt(cand.size()));
	}

	public boolean checkDegree() {
		for (int nod : this.adjacencyList.keySet()) {
			if (this.adjacencyList.get(nod).size() < this.maxDegree[nod - 1])
				return false;
		}
		return true;
	}

	public static RandomGraph uniformFullyConnected(int numNod) {
		RandomGraph ng = new RandomGraph(numNod);
		for (int i = 1; i < numNod; i++) {
			for (int j = i + 1; j <= numNod; j++) {
				ng.setEdge(i, j);
			}
		}
		return ng;

	}

	public static RandomGraph uniformGraph(int numNod, int numEdges, boolean singleComp) {
		if (numEdges < numNod)
			return null;

		RandomGraph ng = new RandomGraph(numNod);

		if (singleComp == true) {
			for (int i = 1; i < numNod; i++)
				ng.setEdge(i, i + 1);
		}

		Random r = new Random();
		while (ng.numEdges < numEdges) {
			int from = 1 + r.nextInt(numNod);
			int to = 1 + r.nextInt(numNod);
			ng.setEdge(from, to);
		}
		return ng;
	}

	public static RandomGraph uniformGraphMaxDegree(int numNod, int numEdges, int maxDegree, boolean singleComp) {
		if (numNod * maxDegree < numEdges)
			return null;
		RandomGraph ng = new RandomGraph(numNod);

		if (singleComp == true) {
			for (int i = 1; i < numNod; i++)
				ng.setEdge(i, i + 1);
		}

		for (int i = 1; i <= numNod; i++) {
			ng.setMaxDegree(i, maxDegree);
		}
		Random r = new Random();
		while (ng.numEdges < numEdges) {
			int from = 1 + r.nextInt(numNod);
			int to = 1 + r.nextInt(numNod);
			ng.setEdge(from, to);
		}
		return ng;
	}

	public static RandomGraph expScaleGraph(int numNod, int maxDegree, boolean singleComp) {
		RandomGraph ng = new RandomGraph(numNod);

		if (singleComp == true) {
			for (int i = 1; i < numNod; i++)
				ng.setEdge(i, i + 1);
		}

		for (int i = 1; i <= numNod; i++) {
			int dnum = (int) (10 * Math.random());
			ng.setMaxDegree(i, (int) (maxDegree * Math.exp(dnum) / Math.exp(10)));
		}
		Random r = new Random();
		while (!ng.checkDegree()) {
			int from = 1 + r.nextInt(numNod);
			int to = 1 + r.nextInt(numNod);
			ng.setEdge(from, to);
		}
		return ng;
	}

}