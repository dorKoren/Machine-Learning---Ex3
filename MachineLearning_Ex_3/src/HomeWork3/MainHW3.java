package HomeWork3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import HomeWork3.Knn.eDistanceCheck;
import HomeWork3.Knn.eMajorityMode;
import weka.core.Instances;

public class MainHW3 {
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	


	public static void main(String[] args) throws Exception {
		
		Instances trainingInstances = loadData("auto_price.txt");
		
		Knn originalDataKnn = new Knn(eDistanceCheck.Regular, trainingInstances);
		
		double OriginalDataCrossValidationError = originalDataKnn.findAndSetBestHyperParameters();
		
		System.out.println("----------------------------");
		System.out.println("Result for original dataset:");
		System.out.println("----------------------------");
		
		System.out.println("Cross validation error with K = " 
		+ originalDataKnn.getKValue() + ", lp = " + originalDataKnn.getPValue()
		+ ", majority function" + "\n= " + originalDataKnn.getMajorityMode().toString()
		+ " for auto_price data is: " + OriginalDataCrossValidationError);
		
		/*----------------------------------------------------------------------------------*/
		
		// Scale the data set.
		FeatureScaler.scaleData(trainingInstances);
		
		Knn scaledDataKnn = new Knn(eDistanceCheck.Regular, trainingInstances);	
		
		double scaledDataCrossValidationError = scaledDataKnn.findAndSetBestHyperParameters();
		
		System.out.println();
		System.out.println();
		System.out.println("----------------------------");
		System.out.println("Result for scaled dataset:");
		System.out.println("----------------------------");
		
		System.out.println("Cross validation error with K = " + 
		scaledDataKnn.getKValue()  + ", lp = " + scaledDataKnn.getPValue() + 
		", majority function\n= " + originalDataKnn.getMajorityMode().name() + 
		" for auto_price data is: " + scaledDataCrossValidationError);
				
		
		/*----------------------------------------------------------------------------------*/
		
		int[] folds = { trainingInstances.numInstances(), 50, 10, 5, 3 };
		
		for (int numOfFolds : folds ) {
			
			// Set the distance check.
			scaledDataKnn.setDistanceCheck(eDistanceCheck.Regular);
			

			// Set the numOfFild member at Knn class 
			// to hold the current number of folds.
			//scaledDataKnn.setNumOfFolds(numOfFolds);
			
			// Partition the data according to the current number of folds.
			scaledDataKnn.partitionTrainingData(numOfFolds);
			
			// Calculate the cross validation error of the regular knn on the dataset.
			scaledDataCrossValidationError = scaledDataKnn.crossValidationError();
		
			System.out.println();
			System.out.println();
			
			
			System.out.println("----------------------------");
			System.out.println("Result for " + numOfFolds +" folds:");
			System.out.println("----------------------------");
			
			System.out.println("Cross validation error of " + scaledDataKnn.getDistanceCheck().name() + " knn on "
					+ "auto_price dataset is " + scaledDataCrossValidationError
					+ " and" + "\n" + "the average elapsed time is " + scaledDataKnn.getTime() / numOfFolds 
					+ "\n" + "The total elapsed time is: " + scaledDataKnn.getTime());
			
			System.out.println();
					
			
			// Set the distance check.
			scaledDataKnn.setDistanceCheck(eDistanceCheck.Efficient);
			

			
			// Calculate the cross validation error of the efficient knn on the dataset.
			scaledDataCrossValidationError = scaledDataKnn.crossValidationError();
						
			System.out.println("Cross validation error of "  + scaledDataKnn.getDistanceCheck().name()  + " knn on "
					+ "auto_price dataset is " + scaledDataCrossValidationError
					+ " and" + "\n" + "the average elapsed time is " + scaledDataKnn.getTime() / numOfFolds
					+ "\n" + "The total elapsed time is: " + scaledDataKnn.getTime());	
						

			
		}

	}

}
