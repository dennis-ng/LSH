import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
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
	public static final int	VECTOR_LENGTH	= 1000;

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

		public void reduce(Text key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			String[] hashVector = conf.get("hashVector").split(",");
			boolean searchPositive = conf.getBoolean("searchPositive",true);
			
			for(Text val : values){
				String[] mappedVector = val.toString().split(",");
				double dot_product=vectorDot(hashVector,mappedVector);
				
				if((dot_product >= 0)==searchPositive)
					context.write(key, val);
			}
		}
	}

	public static void main(String[] args) throws Exception {
		long number_of_neighbours = Long.MAX_VALUE;
		// do{
		Configuration conf = new Configuration();
		StringBuilder hashVector = generateRandomHash();
		conf.set("hashVector", hashVector.toString());
		boolean searchPositive = hashSearch(hashVector,args[2]);
		conf.setBoolean("searchPositive", searchPositive);
		Job job = Job.getInstance(conf);
		job.setJarByClass(FeatureVectorLsh.class);
		job.setMapperClass(TokenizerMapper.class);
		job.setReducerClass(MyReducer.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		if(job.waitForCompletion(true) ){
			number_of_neighbours = job.getCounters().findCounter("org.apache.hadoop.mapred.Task$Counter", "REDUCE_OUTPUT_RECORDS").getValue();
		}else{
			System.exit(1);
		}
		// }while(number_of_neighbours > KNN);
	}

	private static StringBuilder generateRandomHash() {
		StringBuilder hashVector = new StringBuilder();
		for (int i = 1; i < VECTOR_LENGTH; i++) {
			hashVector.append((Math.random() > 0.5)? 1 : -1 ).append(",");
		}
		hashVector.append((Math.random() > 0.5)? 1 : -1);
		return hashVector;
	}

	private static boolean hashSearch(StringBuilder hashVector, String fileName) throws IOException {
		String[] hashArr = hashVector.toString().split(",");
		File file = new File(fileName);
		BufferedReader br = new BufferedReader(new FileReader(file));		
		String[] searchVector = br.readLine().split(","); 
		br.close();
		double dot_product = vectorDot(hashArr, searchVector);
		return (dot_product >= 0);
	}

	private static double vectorDot(String[] hashArr, String[] searchVector) {
		double dot_product=0;
		for(int i = 0;i<searchVector.length;i++){
			dot_product += Double.parseDouble(hashArr[i])*Double.parseDouble(searchVector[i]);
		}
		return dot_product;
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
	 * Calculates the LSH signature of a given vector using the given hash
	 * function. This LSH uses the random project method.
	 * 
	 * @param vect
	 *            The vector to hash.
	 * @param hashFunction
	 *            An array containing normal vectors of random hyperplanes.
	 * @return The signature of the vector with the same length as hashFunction.
	 */
	private static BitSet calculateHash(double[] vect,BitSet[] hashFunction) {
		BitSet signature = new BitSet(hashFunction.length);
		double dot_product = 0;
		for (int i =0;i<hashFunction.length;i++) {
			// The signature contains a set bit(1) if the vector is pointing
			// in the same direction as the normal vector(positive space)
			// and a 0 bit otherwise.
			if (isSameDirection(vect, hashFunction[i]))
				signature.set(i);
		}
		return signature;
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
}
