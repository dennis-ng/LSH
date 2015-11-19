import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.URI;
import java.nio.file.FileAlreadyExistsException;
import java.util.HashSet;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class EuclideanLsh {
	public static int	SIGNATURE_LENGTH	= 20;
	private static int	BUCKET_WIDTH		= 20;
	public static int VECTOR_LENGTH = 61236;
	public static int MULTIPLIER = 100;

	public static class HashSignatureMapper extends Mapper<Object, Text, Text, Text> {

		private float[][]	hashFunction;
		private Text		signature	= new Text();

		@Override
		protected void setup(Mapper<Object, Text, Text, Text>.Context context)
				throws IOException, InterruptedException {
			super.setup(context);
			Configuration conf = context.getConfiguration();
			URI[] uriList = Job.getInstance(conf).getCacheFiles();
			Path filePath = new Path(uriList[0].getPath());
			String configFileName = filePath.getName().toString();
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(configFileName));
			try {
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
			String vect = entry.substring(vectStart + 1);
			StringBuilder stringHash = new StringBuilder();
			for (int i = 0; i < hashFunction.length; i++) {
				stringHash.append(hashed(vect.split("\t"), hashFunction[i]) + ",");
			}
			stringHash.deleteCharAt(stringHash.length() - 1);
			signature.set(stringHash.toString());
			context.write(signature, value);
		}
	}

	private static int hashed(String[] sparseVect, float[] hash) {
		String[] entry;
		double scalar = 0;
		int pos;
		double val;
		for (int i = 0; i < sparseVect.length; i++) {
			entry = sparseVect[i].split(":");
			pos = Integer.parseInt(entry[0]);
			val = Double.parseDouble(entry[1]);
			scalar += val * hash[pos - 1]; // index starts from 1
		}
		return (int) (scalar / (BUCKET_WIDTH));

	}

	private static float[][] generateRandomHash(HashSet<Double[]> usedHashes, int signatureLength) {
		float[][] hashes = new float[signatureLength][VECTOR_LENGTH];
		for (int i = 0; i < signatureLength; i++) {
			for (int j = 0; j < VECTOR_LENGTH; j++) {
				hashes[i][j] = ((float) Math.random() - 0.5f);
			}
			normalize(hashes[i]);
			for (int j = 0; j < VECTOR_LENGTH; j++) {
				hashes[i][j] *= MULTIPLIER;
			}
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

	private static File createConfigFile(float[][] hashFunction, String sigFileName) throws IOException {
		File configFile = new File(sigFileName);
		if (configFile.exists()) { throw new FileAlreadyExistsException(
				"Hash function file already exist. Please use a new name."); }
		configFile.createNewFile();
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(configFile));
		oos.writeObject(hashFunction);
		oos.writeInt(BUCKET_WIDTH);
		oos.close();
		return configFile;
	}

	public static void main(String[] args)
			throws IllegalArgumentException, IOException, ClassNotFoundException, InterruptedException {
		if (args.length < 3) {
			System.err.println(
					"Usage : hadoop jar lsh.jar EuclideanLsh input output save_hash_function_file [-s signature length] [-b bucket width] [-f number of features] [-m multiplier]");
			System.exit(1);
		}
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
					case "-f":
						VECTOR_LENGTH = Integer.parseInt(args[++i]);
						break;
					case "-m":
						MULTIPLIER = Integer.parseInt(args[++i]);
						break;
				}
			}
		}

		HashSet<Double[]> usedHashes = new HashSet<Double[]>();
		float[][] hashFunction = generateRandomHash(usedHashes, SIGNATURE_LENGTH);
		File configFile = createConfigFile(hashFunction, args[2]);
		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf);
		job.addCacheFile(configFile.toURI());
		job.setJarByClass(EuclideanLsh.class);
		job.setMapperClass(HashSignatureMapper.class);
		job.setNumReduceTasks(0);
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
