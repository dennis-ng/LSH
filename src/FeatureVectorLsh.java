import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.net.URI;
import java.util.BitSet;
import java.util.HashSet;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class FeatureVectorLsh {
	public static final int			KNN				= 2;
	public static final int			VECTOR_LENGTH	= 4000;
	public static final int			SKETCH_LENGTH	= 15;
	public static HashSet<BitSet>	usedHashes		= new HashSet<BitSet>();
	public static double			threshold		= 0.5;					// from
																			// -1
																			// to
																			// 1

	public static class TokenizerMapper extends Mapper<Object, Text, BitSetWritable, Text> {

		private Text		classification	= new Text();
		private Text		vector			= new Text();

		private BitSet[]	hashFunction;

		@Override
		protected void setup(Mapper<Object, Text, BitSetWritable, Text>.Context context)
				throws IOException, InterruptedException {
			super.setup(context);
			Configuration conf = context.getConfiguration();
			URI[] uriList = Job.getInstance(conf).getCacheFiles();
			Path filePath = new Path(uriList[0].getPath());
			String configFileName = filePath.getName().toString();
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(configFileName));
			try {
				// this.searchSketch = (BitSet) ois.readObject(); // Skipped
				ois.readObject();
				this.hashFunction = (BitSet[]) ois.readObject();
			}
			catch (ClassNotFoundException e) {
				ois.close();
				throw new IOException("Config file mismatch!");
			}
			finally {
				ois.close();
			}
		}

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String entry = value.toString();
			int vectStart = entry.indexOf("\t");
			String className = entry.substring(0, vectStart);
			String vect = entry.substring(vectStart + 1);
			classification.set(className);

			double[] inputVector = parseDoubleArr(vect.split(","));
			BitSet inputSketch = calculateHash(inputVector, hashFunction);
			BitSetWritable writableSketch = new BitSetWritable();
			writableSketch.set(inputSketch);
			vector.set(vect);
			context.write(writableSketch, value);
		}
	}

	public static class MyReducer extends Reducer<BitSetWritable, Text, DoubleWritable, Text> {
		private BitSet	searchSketch;
		private Text	classification	= new Text();
		private Text	vector			= new Text();

		@Override
		protected void setup(Reducer<BitSetWritable, Text, DoubleWritable, Text>.Context context)
				throws IOException, InterruptedException {
			super.setup(context);
			Configuration conf = context.getConfiguration();
			URI[] uriList = Job.getInstance(conf).getCacheFiles();
			Path filePath = new Path(uriList[0].getPath());
			String configFileName = filePath.getName().toString();
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(configFileName));
			try {
				this.searchSketch = (BitSet) ois.readObject();
			}
			catch (ClassNotFoundException e) {
				ois.close();
				throw new IOException("Config file mismatch!");
			}
			finally {
				ois.close();
			}
		}

		public void reduce(BitSetWritable key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			BitSet hammingMask = (BitSet) searchSketch.clone();
			hammingMask.xor(key.get());
			double hammingDist = hammingMask.cardinality();
			// Similarity range from -1.0 to 1.0
			double similarity = Math.cos((hammingDist / SKETCH_LENGTH) * Math.PI);
			if (similarity >= threshold) {
				DoubleWritable simScore = new DoubleWritable();
				simScore.set(similarity);
				for (Text val : values) {
					// String entry = val.toString();
					// int vectStart = entry.indexOf("\t");
					// String className = entry.substring(0, vectStart);
					// String vect = entry.substring(vectStart + 1);
					// classification.set(className);
					// vector.set(vect);
					// context.write(classification, vector);
					context.write(simScore, val);
				}
			}
		}
	}

	public static class BitSetWritable
			implements Comparable<BitSetWritable>, Writable, WritableComparable<BitSetWritable> {
		private BitSet data;

		public void set(BitSet toSet) {
			this.data = toSet;
		}

		public BitSet get() {
			return this.data;
		}

		@Override
		public String toString() {
			return data.toString();
		}

		@Override
		public void readFields(DataInput dataInput) throws IOException {
			int arrLen = dataInput.readInt();
			byte[] byteDataArr = new byte[arrLen];
			dataInput.readFully(byteDataArr);
			ByteArrayInputStream bis = new ByteArrayInputStream(byteDataArr);
			ObjectInput in = new ObjectInputStream(bis);
			try {
				data = (BitSet) in.readObject();
			}
			catch (ClassNotFoundException e) {
				e.printStackTrace();
			}
			bis.close();
			in.close();
		}

		@Override
		public void write(DataOutput dataOutput) throws IOException {
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			ObjectOutput out = new ObjectOutputStream(bos);
			out.writeObject(data);
			byte[] byteDataArr = bos.toByteArray();
			out.close();
			bos.close();
			dataOutput.writeInt(byteDataArr.length);
			dataOutput.write(byteDataArr);
		}

		@Override
		public int compareTo(BitSetWritable obj) {
			BitSet rhs = ((BitSetWritable) obj).get();
			if (data.equals(rhs))
				return 0;
			if (data.length() != rhs.length())
				return data.length() > rhs.length() ? 1 : -1;
			BitSet xor = (BitSet) data.clone();
			xor.xor(rhs);
			int firstDifferent = xor.length() - 1;
			return data.get(firstDifferent) ? 1 : -1;
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
	private static BitSet calculateHash(double[] vect, BitSet[] hashFunction) {
		BitSet sketch = new BitSet(hashFunction.length);
		double dot_product = 0;
		for (int i = 0; i < hashFunction.length; i++) {
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
		BitSet[] hashFunction = generateRandomHash(usedHashes, SKETCH_LENGTH);
		BitSet searchSketch = calculateHash(searchVector, hashFunction);
		File configFile = createConfigFile(hashFunction, searchSketch);

		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf);
		job.addCacheFile(configFile.toURI());
		job.setJarByClass(FeatureVectorLsh.class);
		job.setMapperClass(TokenizerMapper.class);
		job.setMapOutputKeyClass(BitSetWritable.class);
		// job.setCombinerClass(MyReducer.class);
		job.setReducerClass(MyReducer.class);
		job.setOutputKeyClass(DoubleWritable.class);
		job.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		if (job.waitForCompletion(true)) {
			System.out.println(searchSketch.toString());
			number_of_neighbours = job.getCounters()
					.findCounter("org.apache.hadoop.mapred.Task$Counter", "REDUCE_OUTPUT_RECORDS").getValue();
		}
		else {
			System.exit(1);
		}
	}
}
