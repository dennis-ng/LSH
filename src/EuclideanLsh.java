import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.URI;
import java.util.HashSet;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class EuclideanLsh {
	public static int	SIGNATURE_LENGTH	= 20;
	private static int	BUCKET_WIDTH		= 20;
	public static int VECTOR_LENGTH = 60000;

	public static class HashSignatureMapper extends Mapper<Object, Text, Text, NullWritable> {

		private int[]		searchSig;
		private float[][]	hashFunction;

		@Override
		protected void setup(Mapper<Object, Text, Text, NullWritable>.Context context)
				throws IOException, InterruptedException {
			super.setup(context);
			Configuration conf = context.getConfiguration();
			URI[] uriList = Job.getInstance(conf).getCacheFiles();
			Path filePath = new Path(uriList[0].getPath());
			String configFileName = filePath.getName().toString();
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(configFileName));
			try {
				this.searchSig = (int[]) ois.readObject();
				this.hashFunction = (float[][]) ois.readObject();
			}
			catch (ClassNotFoundException e) {
				ois.close();
				throw new IOException("Config file mismatch!");
			}
			finally {
				ois.close();
			}
		}

		@Override
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String entry = value.toString();
			int vectStart = entry.indexOf("\t");
			String className = entry.substring(0, vectStart);
			String vect = entry.substring(vectStart + 1);

			for (int i = 0; i < hashFunction.length; i++) {
				if (hashed(vect.split("\t"), hashFunction[i]) != searchSig[i]) {
					return;
				}
			}
			context.write(value, NullWritable.get());
		}
	}

	public static class MyReducer extends Reducer<Text, NullWritable, Text, NullWritable> {

		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			for (Text val : values) {
				context.write(key, NullWritable.get());
			}
		}

	}

	private static int hashed(String[] sparseVect, float[] hash) {
		double scalar = 0;
		for (int i = 0; i < sparseVect.length; i++) {
			String[] entry = sparseVect[i].split(":");
			int pos = Integer.parseInt(entry[0]);
			double val = Double.parseDouble(entry[1]);
			scalar += val * hash[pos];
		}
		return (int) (scalar / (BUCKET_WIDTH));

	}

	private static int[] calculateSignature(String[] sparseVect,
			float[][] hashFunction) {
		int sigLength = hashFunction.length;
		int[] signature = new int[sigLength];
		for (int h = 0; h < sigLength; h++) {
			signature[h] = hashed(sparseVect, hashFunction[h]);
		}
		return signature;
	}

	private static float[][] generateRandomHash(HashSet<Double[]> usedHashes, int signatureLength) {
		float[][] hashes = new float[signatureLength][VECTOR_LENGTH];
		for (int i = 0; i < signatureLength; i++) {
			for (int j = 0; j < VECTOR_LENGTH; j++) {
				hashes[i][j] = (float) Math.random() - 0.5f;
			}
			normalize(hashes[i]);
		}
		return hashes;
	}

	private static void normalize(float[] vector) {
		float sumOfSq = 0;
		for (int i = 0; i < vector.length; i++) {
			sumOfSq += vector[i];
		}
		float magnitude = (float) Math.sqrt(sumOfSq);

		for (int i = 0; i < vector.length; i++) {
			vector[i] = vector[i] / magnitude;
		}
	}

	private static File createConfigFile(float[][] hashFunction, int[] searchSig) throws IOException {
		File configFile = File.createTempFile("searchConfig", ".tmp");
		configFile.deleteOnExit();
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(configFile));
		oos.writeObject(searchSig);
		oos.writeObject(hashFunction);
		oos.close();
		return configFile;
	}

	public static void main(String[] args)
			throws IllegalArgumentException, IOException, ClassNotFoundException, InterruptedException {
		if (args.length < 3) {
			System.err.println(
					"Usage : hadoop jar lsh.jar EuclideanLsh input output searchVectorFile [-s signature length] [-b bucket width]");
			System.exit(1);
		}
		BufferedReader br = new BufferedReader(new FileReader(args[2]));
		String searchTerm = br.readLine();
		if (args.length > 3) {
			for (int i = 3; i < args.length; i++) {
				switch (args[i]) {
					case "-sig":
					case "-s":
						SIGNATURE_LENGTH = Integer.parseInt(args[++i]);
						break;
					case "-b":
						BUCKET_WIDTH = Integer.parseInt(args[++i]);
						break;
					case "-l":
						VECTOR_LENGTH = Integer.parseInt(args[++i]);
				}
			}
		}

		HashSet<Double[]> usedHashes = new HashSet<Double[]>();
		float[][] hashFunction = generateRandomHash(usedHashes, SIGNATURE_LENGTH);
		int[] searchSig = calculateSignature(searchTerm.split("\t"),
				hashFunction);
		File configFile = createConfigFile(hashFunction, searchSig);
		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf);
		job.addCacheFile(configFile.toURI());
		job.setJarByClass(CosLsh.class);
		job.setMapperClass(HashSignatureMapper.class);
		job.setCombinerClass(MyReducer.class);
		job.setReducerClass(MyReducer.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(NullWritable.class);
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		if (job.waitForCompletion(true)) {
			System.exit(0);
		}
		else {
			System.exit(1);
		}
	}

}
