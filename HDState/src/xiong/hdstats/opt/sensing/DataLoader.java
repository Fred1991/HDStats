package xiong.hdstats.opt.sensing;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataLoader {

	public static List<double[]> allSensorData(String f) {
		try {
			BufferedReader br = new BufferedReader(new FileReader(f));
			List<double[]> sdata = new ArrayList<double[]>();
			String ln = br.readLine();
			while (ln != null) {
				String[] lns = ln.split(",");
				double[] data = new double[lns.length];
				int index = 0;
				for (String lnn : lns)
					data[index++] = Double.parseDouble(lnn);
				sdata.add(data);
				ln = br.readLine();
			}
			br.close();
			return sdata;
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}

	public static double[] getDataAtTime(List<double[]> sdata, int t) {
		double[] data = new double[sdata.size()];
		for (int i = 0; i < sdata.size(); i++) {
			data[i] = sdata.get(i)[t];
		}
		return data;
	}

	public static double[][] getAllDataBeforeTime(List<double[]> sdata, int t) {
		double[][] data = new double[t][sdata.size()];
	//	System.out.println(data.length+"\t"+data[0].length);
		for (int tt = 0; tt < t; tt++) {
			double[] idata = getDataAtTime(sdata, tt);
			for (int i = 0; i < sdata.size(); i++) {
				data[tt][i] = idata[i];
			}
		}
		return data;
	}
}
