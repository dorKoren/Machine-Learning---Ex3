package HomeWork3;

import java.util.PriorityQueue;
import java.util.Random;

import HomeWork3.Knn.eDistanceCheck;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

class DistanceCalculator {



	/* Public Methods */

	/**
	 * Calculate the distance between two instances. Distance method is defined
	 * in the class member m_pValue: If m_PValue is 0, distance is calculated
	 * using infinity distance, otherwise, distance is calculated using lp
	 * distance.
	 * 
	 * @param instance1
	 * @param instance2
	 * @return Calculated distance.
	 */
	public static double distance(Instance one, Instance two, eDistanceCheck distanceCheck, int pValue, double threshold) {
		
		double distance = 0;

		// Calculate the distance with respect to the given distanceCheck
		// status.
		switch (distanceCheck) {

		case Regular:
			// If p value is 0, we use the specific implementation of
			// infinity-distance.
			distance = pValue != 0 ? lpDisatnce(one, two, pValue) : lInfinityDistance(one, two);
			break;

		default:			
			distance = pValue != 0 ? efficientLpDisatnce(one, two, pValue, threshold)
					: efficientLInfinityDistance(one, two, threshold);
			break;
		}

		return distance;
	}

	/* Private Methods */

	/**
	 * Returns the Lp distance between 2 instances. The lp distance is defined
	 * in the class member m_pValue.
	 * 
	 * @param one
	 * @param two
	 * @return Calculated lp distance.
	 */
	private static double lpDisatnce(Instance one, Instance two, int pValue) {
		double distance = 0;

		// Iterate over all attributes.
		for (int i = 0; i < one.numAttributes(); i++) {

			// If we get to the class index or the id of the attribute,
			// continue.
			if (one.classIndex() == i) {
				continue;
			}

			// Calculate difference of each attribute in abs value, and raise it
			// to P.
			distance += Math.pow(Math.abs(one.value(i) - two.value(i)), pValue);
		}

		// Return the P root of the calculated sum.
		return Math.pow(distance, (1.0 / pValue));
	}

	/**
	 * Returns the L infinity distance between 2 instances.
	 * 
	 * @param one
	 * @param two
	 * @return
	 */
	private static double lInfinityDistance(Instance one, Instance two) {
		double maxDistance = 0;

		// Iterate over all attributes.
		for (int i = 0; i < one.numAttributes(); i++) {
			if (one.classIndex() == i) {
				// Ignore the class and 'id' attributes.
				continue;
			}

			double realDistance = Math.abs(one.value(i) - two.value(i));

			// If we find value that larger than the max distance, change it.
			if (realDistance > maxDistance) {
				maxDistance = realDistance;
			}
		}

		// Return the max distance with respect to the abs value.
		return maxDistance;
	}

	/**
	 * Returns the Lp distance between 2 instances, while using an efficient
	 * distance check.
	 * 
	 * @param one
	 * @param two
	 * @return
	 */
	private static double efficientLpDisatnce(Instance one, Instance two, int pValue, double threshold) {
		
		double distance = 0;
		
		// Iterate over all attributes.
		
		for (int i = 0; i < one.numAttributes(); i++) {
			// If we get to the class index or the id of the attribute,
			// continue.
			if (one.classIndex() == i) {
				continue;
			}
			
			// Calculate difference of each attribute in abs value, and raise it
			// to P.
			distance += Math.pow(Math.abs(one.value(i) - two.value(i)), pValue);
			
			if (Math.pow(threshold, pValue) < distance) {
				break;
			}
		}
		
		return Math.pow(distance, (1.0 / pValue));
	}

	/**
	 * Returns the Lp distance between 2 instances, while using an efficient
	 * distance check.
	 * 
	 * @param one
	 * @param two
	 * @return
	 */
	private static double efficientLInfinityDistance(Instance one, Instance two, double threshold) {
		
		double maxDistance = 0;
		
		// Iterate over all attributes.
		for (int i = 0; i < one.numAttributes(); i++) {	
			if (one.classIndex() == i) {
				// Ignore the class and 'id' attributes.
				continue;
			}
			
			double realDistance = Math.abs(one.value(i) - two.value(i));
			
			if (threshold < realDistance) {       
				break;
			}
						
			// If we find value that larger than the max distance, change it.
			if (realDistance > maxDistance) {
				maxDistance = realDistance;
			}
		}
		
		// Return the max distance with respect to the abs value.
		return maxDistance;
	}
}

public class Knn implements Classifier {

	/* Inner class */

	/**
	 * A neighbor class storing an instance, and its distance from the
	 * classified instance.
	 */
	private class Neighbor implements Comparable<Neighbor> {
		Instance instance;
		double distance;

		public Neighbor(Instance instance, double distance) {
			this.instance = instance;
			this.distance = distance;
		}

		@Override
		/**
		 * Return 1 for larger neighbors, -1 for smaller neighbors, and 0
		 * otherwise.
		 */
		public int compareTo(Neighbor otherNeighbor) {
			double diff = otherNeighbor.distance - this.distance;

			if (diff > 0) {
				return 1;
			}

			if (diff < 0) {
				return -1;
			}

			return 0;
		}
	}

	/* Class Members */

	/**
	 * Random object for randomizing training data
	 */
	private Random m_random = new Random();

	/**
	 * Phase 2 classifications time counter. Advanced by classifications invoked
	 * from calcAvgError only! This ensures we are not measuring classifications
	 * of the editing functions.
	 */
	private long m_timeCount = 0;


	/* General Members */
	public enum eDistanceCheck {
		Regular, Efficient
	}

	private eDistanceCheck m_distanceCheck;
	private Instances m_trainingInstances;
	private double m_threshold; 
	private Neighbor[] m_neighbors;

	/* Hyper Parameters */
	private int m_pValue;
	private int m_kValue;

	public enum eMajorityMode {
		Uniform, Weighted
	};

	private eMajorityMode m_majorityMode;

	/* Cross Validation Members */
	private int m_numOfFolds;
	private Instances[] m_partitionedTrainingData;
	private Instances m_testingInstances;

	/* Constructor */
	public Knn(eDistanceCheck distanceCheck, Instances trainingInstances) {
		this.m_distanceCheck = distanceCheck;
		this.m_trainingInstances = trainingInstances;
	}

	/* Getters & setters */
	public eDistanceCheck getDistanceCheck() {
		return this.m_distanceCheck;
	}

	public void setDistanceCheck(eDistanceCheck distanceCheck) {
		this.m_distanceCheck = distanceCheck;
	}

	public int getPValue() {
		return this.m_pValue;
	}

	public int getKValue() {
		return this.m_kValue;
	}

	public eMajorityMode getMajorityMode() {
		return this.m_majorityMode;
	}

	public void setNumOfFolds(int numOfFolds) {
		this.m_numOfFolds = numOfFolds;
	}

	public void setThreshold(double threshold) {
		this.m_threshold = threshold;
	}

	public double getThreshold() {
		return this.m_threshold;
	}

	public void setNeighbors(Neighbor[] neighbor) {
		this.setNeighbors(neighbor);  

	}
	
	public long getTime(){
		return m_timeCount;
	}
	
	public void setKValue(int kValue) {
		this.m_kValue = kValue;
	}
	
	public void setPValue(int pValue) {
		this.m_pValue = pValue;
	}
 	
	public void setMajorityMode(eMajorityMode majorityMode) {
		this.m_majorityMode = majorityMode;
	}
	

	
	

	/* Functions */

	@Override
	/**
	 * Build the knn classifier. In our case, simply stores the given instances
	 * for later use in the prediction.
	 * 
	 * @param instances
	 */
	public void buildClassifier(Instances instances) throws Exception {

		this.m_trainingInstances = new Instances(instances);
	}

	/**
	 * Partition the (shuffled) training data into numOfFolds different
	 * partitions. Partitions are stored in m_partitionedTrainingData array.
	 * 
	 * @param numOfFolds
	 *            Number of partitions to create.
	 */
	public void partitionTrainingData(int numOfFolds) { 
		// Copy training data locally.
		Instances trainingData = new Instances(this.m_trainingInstances);

		// Create a new folds array
		this.m_partitionedTrainingData = new Instances[numOfFolds];
		this.m_numOfFolds = numOfFolds;

		// Initialize every fold with a new instances object.
		for (int i = 0; i < numOfFolds; i++) {
			this.m_partitionedTrainingData[i] = new Instances(trainingData, 0);
		}

		// Shuffle training data for more reliable cross validation.
		trainingData.randomize(this.m_random);

		// Partition the training data into the different folds.
		for (int i = 0; i < trainingData.numInstances(); i++) {
			this.m_partitionedTrainingData[i % numOfFolds].add(trainingData.instance(i));
		}
	}

	/**
	 * Fold the partitioned training data. Generating training and testing data
	 * out of the partitioned data. Fold 'foldIndex' will be used for testing.
	 * The rest of the partitions will be used for training. This means taking
	 * m_instancesFolds[foldIndex] as testing, and the rest of the partitions
	 * are combined into training data.
	 * 
	 * @param foldIndex
	 */
	private void foldPartitionedData(int foldIndex) { 
		// Clear the global training set of instances.
		this.m_trainingInstances = new Instances(this.m_trainingInstances, 0);

		// Use foldIndex fold as testing.
		this.m_testingInstances = this.m_partitionedTrainingData[foldIndex];

		// Iterate through all folds.
		for (int i = 0; i < this.m_numOfFolds; i++) {
			// Use all instances except foldIndex as training.
			if (i != foldIndex) {
				// Add all instances from partition i to the training set.
				for (int j = 0; j < this.m_partitionedTrainingData[i].numInstances(); j++) {
					this.m_trainingInstances.add(this.m_partitionedTrainingData[i].instance(j));
				}
			}
		}
	}

	/**
	 * Auxiliary function to find and set best hyper parameters. Parameters are:
	 * - K (1-20): Number of nearest neighbors. - P (0-3): The p parameter of
	 * the l-p distance function. - majorityMode (uniform, weighted): Whether
	 * distances of nearest neighbors matter.
	 * 
	 * @return Cross validation error of best found hyper parameters.
	 */
	public double findAndSetBestHyperParameters() {
		// Create 10 folds partitions out of the training data
		partitionTrainingData(10);

		// Iterate through all possible hyper-parameters,
		// Finding the best combination.
		int bestKValue = 0;
		int bestPValue = 0;
		eMajorityMode bestMajorityMode = eMajorityMode.Uniform;
		double bestCrossValidationError = Double.MAX_VALUE;
		double bestThreshold = 0; 

		for (int kValue = 1; kValue <= 20; kValue++) {
			for (int pValue = 0; pValue <= 3; pValue++) {
				for (eMajorityMode majorityMode : eMajorityMode.values()) {
					// Set global hyper parameters for the current test.
					this.m_kValue = kValue;
					this.m_pValue = pValue;
					this.m_majorityMode = majorityMode;
					
					// Calculate cross validation error
					double crossValidationError = this.crossValidationError(); 
																				
					// We found better set of hyper parameters
					// Update our findings
					if (crossValidationError < bestCrossValidationError) {
						bestKValue = kValue;
						bestPValue = pValue;
						bestMajorityMode = majorityMode;
						bestCrossValidationError = crossValidationError;
						bestThreshold = findMaxDistance(this.m_neighbors); // <---------------------------------tami
					}
				}
			}
		}

		// Set best found hyper parameters
		this.m_kValue = bestKValue;
		this.m_pValue = bestPValue;
		this.m_majorityMode = bestMajorityMode;
		this.m_threshold = bestThreshold;
			
		// Return the best crossValidationError found during the search
		return bestCrossValidationError;
	}

	/**
	 * 
	 * @param neighbors
	 * @return
	 */
	private double findMaxDistance(Neighbor[] neighbors) {

		double maxNeighborDistanse = -1;

		// Iterate over all the k neighbors and find the neighbor
		// with thw max distance.
		for (Neighbor currentNeighbor : neighbors) {

			maxNeighborDistanse = maxNeighborDistanse > currentNeighbor.distance ? maxNeighborDistanse : currentNeighbor.distance;
		}
		
		return (double) maxNeighborDistanse;
	}

	/**
	 * Returns the knn prediction on the given instance.
	 * 
	 * @param instance
	 * @return The instance predicted value.
	 */
	public double regressionPrediction(Instance instance) {

		// On input instance x find k nearest neighbors {x(i)}
		// for i ? {1,...,k}.
		Neighbor[] neighbors = findNearestNeighbors(instance);
		
		this.m_neighbors = neighbors; 

		return m_majorityMode.equals(eMajorityMode.Uniform) ? this.getAverageValue(neighbors)
				: this.getWeightedAverageValue(neighbors);
	}

	/**
	 * Caclcualtes the average error on a give set of instances. The average
	 * error is the average absolute error between the target value and the
	 * predicted value across all insatnces.
	 * 
	 * @param insatnces
	 * @return average error.
	 */
	public double calcAvgError(Instances insatnces) {
		long initTime = System.nanoTime();  // Take initial time measurement.
		double sum = 0;
		m_timeCount=0;
		
		// Iterate across all instances end sum the abs error between the
		// target value and the predicted value.
		for (Instance instance : insatnces) {
			double targetValue = instance.classValue();
			double predictedValue = this.regressionPrediction(instance);
			sum += Math.abs(targetValue - predictedValue); 
		}
		
		// Add execution time to time count
		this.m_timeCount += (System.nanoTime() - initTime); 
		return (double) sum / insatnces.numInstances();
	}

	/**
	 * Calculate the cross validation error. This is the average error on each
	 * fold acting as testing, against all other folds acting as training.
	 * 
	 * @return calculated error.
	 */
	public double crossValidationError() {
		
		// Backup the global training set of instances.
		Instances trainingInstacesBackup = this.m_trainingInstances;

		// Iterate through all folds
		double crossValidationError = 0;
		for (int i = 0; i < this.m_numOfFolds; i++) {
			// Prepare fold testing and training sets data.
			this.foldPartitionedData(i);

			// Calculate fold error
			crossValidationError += this.calcAvgError(this.m_testingInstances);
		}

		// Restore the global training data field.
		this.m_trainingInstances = trainingInstacesBackup;

		return (double) crossValidationError / this.m_numOfFolds;
	}

	/**
	 * Finds the k nearest neighbors.
	 * 
	 * @param instance
	 */
	public Neighbor[] findNearestNeighbors(Instance instance) {

		// Create a priority queue to store the K nearest neighbors.
		PriorityQueue<Neighbor> kNearestNeighbors = new PriorityQueue<Neighbor>(this.m_kValue);

		// Iterate through all the training instances.
		for (int i = 0; i < this.m_trainingInstances.numInstances(); i++) {
			// Create a neighbor object.
			Instance neighborInstance = this.m_trainingInstances.instance(i);
			double distance = DistanceCalculator.distance(instance, neighborInstance, this.getDistanceCheck(), this.getPValue(), this.getThreshold()); 
			
			Neighbor neighbor = new Neighbor(neighborInstance, distance);

			// Add neighbor to the K nearest neighbors queue.
			kNearestNeighbors.add(neighbor);

			// Make sure the K nearest neighbors queue does not exceed K
			// neighbors.
			if (kNearestNeighbors.size() > this.m_kValue) {
				// Remove the largest neighbor.
				kNearestNeighbors.poll();
			}
		}
		
		// Return the K nearest neighbors as an array.
		return kNearestNeighbors.toArray(new Neighbor[kNearestNeighbors.size()]);
	}

	/**
	 * Cacluates the average value of the given elements in the collection.
	 * 
	 * @param
	 * @return
	 */
	public double getAverageValue(Neighbor[] kNearestNeighbors) {

		double instancesClassValues = 0;

		// Iterate through all the k neighbors and sum their class value.
		for (Neighbor neighbor : kNearestNeighbors) {
			instancesClassValues += neighbor.instance.classValue();
		}

		// Return the prediction.
		return (double) instancesClassValues / kNearestNeighbors.length;
	}

	/**
	 * Calculates the weighted average of the target values of all the elements
	 * in the collection with respect to their distance from a specific
	 * instance.
	 * 
	 * @param kNearestNeighbors
	 *            Array of neighbors.
	 * @return Most prevalent class value.
	 */
	public double getWeightedAverageValue(Neighbor[] kNearestNeighbors) {
		
		double weightedAverageValue = 0;

		// Iterate through the k nearest neighbors.
		for (Neighbor neighbor : kNearestNeighbors) {

			double neighborClassValue = neighbor.instance.classValue();
			
			if (neighbor.distance == 0) {
				weightedAverageValue += neighborClassValue;
			}
			
			double neighborDistance = (double) 1.0 / Math.pow(neighbor.distance, 2);	
			double numerator = (double) neighborDistance * neighborClassValue; 
			double denominator = neighborDistance;

			weightedAverageValue += numerator / denominator;
		}

		return (double) weightedAverageValue;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// TODO Auto-generated method stub - You can ignore.
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub - You can ignore.
		return null;
	}

	@Override
	public double classifyInstance(Instance instance) {
		// TODO Auto-generated method stub - You can ignore.
		return 0.0;
	}

}