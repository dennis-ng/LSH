package knn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;

public class LazyKnnSearch {
	private float[][] hashFunction;
	private int bucketWidth;
	private String[] searchSignature;
	private ArrayList<DistanceEntryPair> knnDistances = new ArrayList<DistanceEntryPair>();

	public LazyKnnSearch(String hashFile, String searchFile)
			throws ClassNotFoundException, IOException {
		loadHashFunction(hashFile);
		hashSearchFile(searchFile);
	}

	private void hashSearchFile(String searchFile) throws IOException {
		// Read in and hash search vector
		BufferedReader br = new BufferedReader(new FileReader(new File(
				searchFile)));
		String[] searchVector = br.readLine().split("\t");
		br.close();
		this.searchSignature = calculateSignature(searchVector);
	}

	private int hashed(String[] sparseVect, float[] hash, int bucketWidth) {
		double scalar = 0;
		for (int i = 0; i < sparseVect.length; i++) {
			String[] entry = sparseVect[i].split(":");
			int pos = Integer.parseInt(entry[0]);
			double val = Double.parseDouble(entry[1]);
			scalar += val * hash[pos - 1];
		}
		return (int) (scalar / bucketWidth);

	}

	public String[] calculateSignature(String[] searchVector) {
		int sigLength = hashFunction.length;
		String[] signature = new String[sigLength];
		for (int h = 0; h < sigLength; h++) {
			signature[h] = ""
					+ hashed(searchVector, hashFunction[h], bucketWidth);
		}
		return signature;
	}

	public void loadHashFunction(String fileName) throws IOException,
	ClassNotFoundException {
		// Read in hash function
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(
				new File(fileName)));
		this.hashFunction = (float[][]) ois.readObject();
		this.bucketWidth = ois.readInt();
		ois.close();
	}

	public void searchInFile(Path path) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(
				path.toUri())));
		String lshEntry;
		while ((lshEntry = br.readLine()) != null) {
			int dist = getHammingDistance(getSignature(lshEntry));
			knnDistances.add(new DistanceEntryPair(dist, getEntry(lshEntry)));
		}
	}


	private int getHammingDistance(String[] entry) {
		int dist = 0;
		for (int i = 0; i < searchSignature.length; i++) {
			if (!searchSignature[i].equals(entry[i])) {
				dist++;
			}
		}
		return dist;
	}

	private String[] getSignature(String lshEntry) {
		return lshEntry.substring(0, lshEntry.indexOf('\t'))
				.split(",");
	}

	private String getEntry(String lshEntry) {
		return lshEntry
				.substring(lshEntry.indexOf('\t') + 1, lshEntry.length());
	}


	// Writes k number of nearest neighbours into the output file
	public void getNeighbours(int k, File output) throws IOException {
		Collections.sort(knnDistances);
		PrintWriter pw = new PrintWriter(new FileWriter(output));
		for(int i=0;i<k;i++) {
			pw.print(knnDistances.get(i).distance + "\t");
			pw.println(knnDistances.get(i).entry);
		}
		pw.close();
	}
	private static class DistanceEntryPair implements
	Comparable<DistanceEntryPair> {
		public int distance;
		public String entry;

		public DistanceEntryPair(int distance, String entry) {
			this.distance = distance;
			this.entry = entry;
		}

		@Override
		public int compareTo(DistanceEntryPair other) {
			return this.distance - other.distance;
		}
	}


}
