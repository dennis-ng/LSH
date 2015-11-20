package knn;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class KnnDriver {

	private static final String LSH_SIG_Directory = "Test/Test";

	public static void main(String[] args) throws ClassNotFoundException,
	IOException {
		int knn = 0;
		String searchFile = null;
		String outputFileName = null;
		;
		String hashFile = null;
		if (args.length < 4) {
			// complain
		} else {
			for (int i = 0; i < args.length; i++) {
				switch (args[i]) {
					case "-k":
					case "-knn":
						knn = Integer.parseInt(args[++i]);
						break;
					case "-f":
						searchFile = (args[++i]);
						break;
					case "-o":
						outputFileName = (args[++i]);
						break;
					case "-h":
						hashFile = (args[++i]);
						break;
				}
			}
		}
		LazyKnnSearch searcher = new LazyKnnSearch(hashFile, searchFile);
		Files.walk(Paths.get(LSH_SIG_Directory)).forEach(filePath -> {
			try {
				if (Files.isRegularFile(filePath)
						&& !Files.isHidden(filePath)) {
					searcher.searchInFile(filePath);
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		});
		File outputFile = new File(outputFileName);
		if (!outputFile.exists()) {
			outputFile.createNewFile();
		}
		searcher.getNeighbours(knn, outputFile);
	}
}
