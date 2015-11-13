import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.URI;
import java.util.BitSet;
import java.util.HashSet;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class FeatureVectorLsh {
	public static final int KNN = 2;
	public static final int			VECTOR_LENGTH	= 4000;
	public static HashSet<BitSet>	usedHashes		= new HashSet<BitSet>();

	public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {

		private Text						classification	= new Text();
		private Text vector = new Text();

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String entry = value.toString();
			int vectStart = entry.indexOf("\t");
			String className = entry.substring(0,vectStart);
			String vect = entry.substring(vectStart+1);
			classification.set(className);
			vector.set(vect);
			context.write(classification, vector);
		}
	}

	public static class MyReducer extends Reducer<Text, Text, Text, Text> {
		private BitSet		searchSketch;
		private BitSet[]	hashFunction;
		@Override
		protected void setup(Reducer<Text, Text, Text, Text>.Context context) throws IOException, InterruptedException {
			// TODO Auto-generated method stub
			super.setup(context);
			Configuration conf = context.getConfiguration();
			URI[] uriList = Job.getInstance(conf).getCacheFiles();
			Path filePath = new Path(uriList[0].getPath());
			String configFileName = filePath.getName().toString();
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(configFileName));
			try {
				this.searchSketch = (BitSet) ois.readObject();
				this.hashFunction = (BitSet[]) ois.readObject();
			}
			catch (ClassNotFoundException e) {
				throw new IOException("Config file mismatch!");
			}
			ois.close();
		}

		public void reduce(Text key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			
			for(Text val : values){
				double[] mappedVector = parseDoubleArr(val.toString().split(","));
				boolean matched = true;
				for (int i = 0; i < hashFunction.length; i++) {
					if (isSameDirection(mappedVector, hashFunction[i]) != searchSketch.get(i)) {
						// Differ from the search sketch, can already ignore
						// this mappedVector
						matched = false;
						break;
					}
				}
				if (matched) {
					// Write the context if the sketch of the value is same
					// as
					// the search sketch
					context.write(key, val);
				}
			}
		}
	}

	private static File createConfigFile(BitSet[] hashFunction, BitSet searchSketch)
			throws IOException, FileNotFoundException {
		File configFile = File.createTempFile("searchConfig", ".tmp");
		configFile.deleteOnExit();
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(configFile));
		oos.writeObject(searchSketch);
		oos.writeObject(hashFunction);
		oos.close();
		return configFile;
	}

	private static BitSet[] generateRandomHash(HashSet<BitSet> generatedHistory, int numOfNewHash) {
		int i = 0;
		BitSet[] newHashFunction = new BitSet[numOfNewHash];
		while (i < numOfNewHash) {
			BitSet newHash = new BitSet(VECTOR_LENGTH);
			for (int b = 0; b < VECTOR_LENGTH; b++) {
				if (Math.random() > 0.5)
					newHash.set(b);
			}
			boolean success = generatedHistory.add(newHash);
			if (success) {
				newHashFunction[i] = newHash;
				i++;
			}
		}
		return newHashFunction;
	}

	/**
	 * Calculates the LSH sketch of a given vector using the given hash
	 * function. This LSH uses the random project method.
	 * 
	 * @param vect
	 *            The vector to hash.
	 * @param hashFunction
	 *            An array containing normal vectors of random hyperplanes.
	 * @return The sketch of the vector with the same length as hashFunction.
	 */
	private static BitSet calculateHash(double[] vect,BitSet[] hashFunction) {
		BitSet sketch = new BitSet(hashFunction.length);
		double dot_product = 0;
		for (int i =0;i<hashFunction.length;i++) {
			// The sketch contains a set bit(1) if the vector is pointing
			// in the same direction as the normal vector(positive space)
			// and a 0 bit otherwise.
			if (isSameDirection(vect, hashFunction[i]))
				sketch.set(i);
		}
		return sketch;
	}

	/**
	 * 
	 * @param vect
	 *            The vector to be compared.
	 * @param normalVect
	 *            Each true bit in the normalVect BitSet represents a value of
	 *            +1 , and each false bit represents a value of -1
	 * @return true if the vector is pointing in the same direction as the
	 *         normal vector(positive space), false otherwise
	 */
	private static boolean isSameDirection(double[] vect, BitSet normalVect) {
		double dot_product = 0;
		// Each true bit represents +1 in normal vector value, and each
		// false bit represents a -1
		for (int b = 0; b < VECTOR_LENGTH; b++) {
			if (normalVect.get(b)) {
				dot_product += vect[b];
			}
			else {
				dot_product -= vect[b];
			}
		}
		return dot_product >= 0;
	}
	
	private static double[] parseDoubleArr(String[] strArr) {
		int len = strArr.length;
		double[] parsedArr = new double[len];
		for (int i = 0; i < len; i++) {
			parsedArr[i] = Double.parseDouble(strArr[i]);
		}
		return parsedArr;
	}

	public static void main(String[] args) throws Exception {
		long number_of_neighbours = Long.MAX_VALUE;
		if (args.length < 3) {
			System.err.println("Usage : hadoop jar lsh.jar FeatureVectorLsh input output searchVectorFile");
			System.exit(1);
		}
		BufferedReader br = new BufferedReader(new FileReader(args[2]));
		String searchTerm = br.readLine();
		double[] searchVector = parseDoubleArr(searchTerm.split(","));
		BitSet[] hashFunction = generateRandomHash(usedHashes, 10);
		BitSet searchSketch = calculateHash(searchVector, hashFunction);
		File configFile = createConfigFile(hashFunction, searchSketch);
	
		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf);
		job.addCacheFile(configFile.toURI());
		job.setJarByClass(FeatureVectorLsh.class);
		job.setMapperClass(TokenizerMapper.class);
		job.setReducerClass(MyReducer.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		if(job.waitForCompletion(true) ){
			System.out.println(searchSketch.toString());
			number_of_neighbours = job.getCounters().findCounter("org.apache.hadoop.mapred.Task$Counter", "REDUCE_OUTPUT_RECORDS").getValue();
		}else{
			System.exit(1);
		}
	}
}
