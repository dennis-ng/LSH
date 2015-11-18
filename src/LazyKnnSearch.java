import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;

public class LazyKnnSearch {

	public static void main(String[] args) {
		int KNN = 20;
		if (args.length > 0) {
			for (int i = 0; i < args.length; i++) {
				switch (args[i]) {
					case "-k":
					case "-knn":
						KNN = Integer.parseInt(args[++i]);
						break;
				}
			}
		}
		ArrayList<DistanceIndexPair> knnDistances = new ArrayList<DistanceIndexPair>();
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(new File("small_hash_func")));
			float[][] hashFunction = (float[][]) ois.readObject();
			int bucketWidth = ois.readInt();
			ois.close();
			BufferedReader br = new BufferedReader(new FileReader(new File("searchVector")));
			String[] searchVector = br.readLine().split("\t");
			String[] searchSig = calculateSignature(searchVector, hashFunction, bucketWidth);

			br = new BufferedReader(new FileReader(new File("output")));
			String lshEntry;
			int i = 0;
			while ((lshEntry = br.readLine()) != null) {
				String[] storedSig = lshEntry.substring(0, lshEntry.indexOf('\t')).split(",");
				knnDistances.add(new DistanceIndexPair(getHammingDistance(storedSig, searchSig), i++));
			}
			br.close();
			Collections.sort(knnDistances);
			ArrayList<Integer> lineNumbers = new ArrayList<Integer>(KNN);
			for (int k = 0; k < KNN; k++) {
				lineNumbers.add(knnDistances.get(i).index);
			}
			Collections.sort(lineNumbers);

			// Read the input file again, this time retrieve the whole entry
			// that matches KNN and write to output
			i = 0;
			int a = 0;
			br = new BufferedReader(new FileReader(new File("output")));
			PrintWriter pw = new PrintWriter(new FileWriter("knn.txt"));
			while ((lshEntry = br.readLine()) != null && a < KNN) {
				if (lineNumbers.get(a) == i) {
					pw.println(lshEntry.substring(lshEntry.indexOf('\t') + 1));
					a++;
				}
				i++;
			}
			pw.close();
		}
		catch (IOException e) {
			e.printStackTrace();
		}
		catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
	}

	private static int getHammingDistance(String[] storedSig, String[] searchSig) {
		int dist = 0;
		for (int i = 0; i < searchSig.length; i++) {
			if (searchSig[i] != storedSig[i])
				dist++;
		}
		return dist;
	}

	private static int hashed(String[] sparseVect, float[] hash, int bucketWidth) {
		double scalar = 0;
		for (int i = 0; i < sparseVect.length; i++) {
			String[] entry = sparseVect[i].split(":");
			int pos = Integer.parseInt(entry[0]);
			double val = Double.parseDouble(entry[1]);
			scalar += val * hash[pos];
		}
		return (int) (scalar / (bucketWidth));

	}

	private static String[] calculateSignature(String[] sparseVect, float[][] hashFunction, int bucketWidth) {
		int sigLength = hashFunction.length;
		String[] signature = new String[sigLength];
		for (int h = 0; h < sigLength; h++) {
			signature[h] = "" + hashed(sparseVect, hashFunction[h], bucketWidth);
		}
		return signature;
	}

	private static class DistanceIndexPair implements Comparable<DistanceIndexPair> {
		public int	distance;
		public int	index;

		public DistanceIndexPair(int distance, int index) {
			this.distance = distance;
			this.index = index;
		}

		@Override
		public int compareTo(DistanceIndexPair other) {
			return this.distance - other.distance;
		}
	}
}
